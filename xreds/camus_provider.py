"""
# Extract, Load and Build Kerchunk aggregations on the fly from hierarchical metadata and kerchunk indexes.

MIT License Copyright (c) 2024 Camus Energy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
import json
import datetime
from enum import unique, Enum
from typing import Optional, Any, Iterable
import os
import ujson
import itertools
import copy

import dask.array as da

import base64

import gzip
import pandas as pd
from google.cloud import bigquery

import xarray as xr
import numpy as np
import fsspec
import datatree

from xpublish import Plugin, hookimpl

from .config import settings


logger = logging.getLogger("uvicorn")

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)


def query_job(
        axes: list[pd.Index],
        variables: list[str],
        table_name: str,
) -> (str, bigquery.QueryJobConfig):
    """
    Construct a query string for the variable name(s) to be used in the fetcher.
    :return: a big query job to serach for the variable index metadata
    """
    logger.info("Querying table: %s", table_name)
    # varname example e.g. t2m/instant/heightAboveGround
    keys = ["varname", "stepType", "typeOfLevel"]
    mapped_query_parts = [
        {k: v for k, v in zip(keys, fullvar.split("/"))} for fullvar in variables
    ]

    var_where_clause = " OR ".join(
        [
            f"(varname = @varname_{v_cnt:03d} AND stepType = @stepType_{v_cnt:03d} AND typeOfLevel = @typeOfLevel_{v_cnt:03d})"
            for v_cnt in range(len(variables))
        ]
    )

    time_clause_parts = []
    time_clause_params = {}
    # Big Query BETWEEN is inclusive, so we can use the min and the max of the axes to filter the data
    for ax in axes:
        if ax.name == "step" and ax.dtype == object:
            # For horizon mode, the step index is an array of time delta ranges
            lb = ax.map(min).min().to_timedelta64().astype("int64")
            ub = ax.map(max).max().to_timedelta64().astype("int64")
            time_clause_parts.append(f"step BETWEEN @step_lb AND @step_ub")
            time_clause_params["step_lb"] = int(lb)
            time_clause_params["step_ub"] = int(ub)

        else:
            name = ax.name
            time_clause_parts.append(f"{name} BETWEEN @{name}_lb AND @{name}_ub")
            time_clause_params[f"{name}_lb"] = int(ax.min().to_numpy().astype("int64"))
            time_clause_params[f"{name}_ub"] = int(ax.max().to_numpy().astype("int64"))

    time_where_clause = " AND ".join(time_clause_parts)

    # Optimize the selected columns later. Consider debug value of keeping the metadata?
    query = f"SELECT * FROM `{table_name}` WHERE ({time_where_clause}) AND ({var_where_clause})"

    variable_query_parameters = [
        bigquery.ScalarQueryParameter(f"{k}_{v_cnt:03d}", "STRING", v)
        for v_cnt, parms in enumerate(mapped_query_parts)
        for k, v in parms.items()
    ]

    variable_query_parameters.extend(
        [
            bigquery.ScalarQueryParameter(k, "INT64", v)
            for k, v in time_clause_params.items()
        ]
    )
    return query, bigquery.QueryJobConfig(query_parameters=variable_query_parameters)


def get_kerchunk_index(
        axes: list[pd.Index],
        variables: list[str],
        table_name: str,
) -> pd.DataFrame:
    """
    Get the kerchunk index for the given variables and NODD model.
    This method is cached for the life of the process.
    :param variables: list of variable names
    :param axes: list of pd.Index objects for the time aggregation axes of the dataset
    :param table_name: the table
    :return: the kerchunk index for the given variables and model
    """
    query, job_config = query_job(axes, variables, table_name)
    bq_client = bigquery.Client(location="us-central1")
    job = bq_client.query(query, job_config=job_config)
    return (
        job.result()
        .to_dataframe()
        .astype(
            dict(
                step="timedelta64[ns]",
                valid_time="datetime64[ns]",
                time="datetime64[ns]",
                # Since this is a cast not a parse, decode the index metadata for better debugging
                grib_updated_at="datetime64[ns]",
                idx_updated_at="datetime64[ns]",
                indexed_at="datetime64[ns]",
            )
        )
    )


COORD_DIM_MAPPING: dict[str, str] = dict(
    time="run_times",
    valid_time="valid_times",
    step="model_horizons",
)


def build_path(path: Iterable[str | None], suffix: Optional[str] = None):
    """
    Returns the path without a leading "/"
    :param path: the path elements which may include None
    :param suffix: a last element if any
    :return: the path as a string
    """
    return "/".join([val for val in [*path, suffix] if val is not None]).lstrip("/")


def repeat_steps(step_index: pd.TimedeltaIndex, to_length: int) -> np.array:
    return np.tile(step_index.to_numpy(), int(np.ceil(to_length / len(step_index))))[
           :to_length
           ]


def create_steps(steps_index: pd.Index, to_length) -> np.array:
    return np.vstack([repeat_steps(si, to_length) for si in steps_index])


def store_coord_var(key: str, zstore: dict, coords: tuple[str, ...], data: np.array):
    if np.isnan(data).any():
        if f"{key}/.zarray" not in zstore:
            logger.debug("Skipping nan coordinate with no variable %s", key)
            return
        else:
            logger.info("Trying to add coordinate var %s with nan value!", key)

    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])
    # Use list not tuple
    zarray["chunks"] = [*data.shape]
    zarray["shape"] = [*data.shape]
    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    vkey = ".".join(["0" for _ in coords])
    data_bytes = data.tobytes()
    try:
        enocded_val = data_bytes.decode("ascii")
    except UnicodeDecodeError:
        enocded_val = (b"base64:" + base64.b64encode(data_bytes)).decode("ascii")
    zstore[f"{key}/{vkey}"] = enocded_val


def store_data_var(
        key: str,
        zstore: dict,
        dims: dict[str, int],
        coords: dict[str, tuple[str, ...]],
        data: pd.DataFrame,
        steps: np.array,
        times: np.array,
        lvals: Optional[np.array],
):
    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])

    dcoords = coords["datavar"]

    # The lat/lon y/x coordinates are always the last two
    lat_lon_dims = {
        k: v for k, v in zip(zattrs["_ARRAY_DIMENSIONS"][-2:], zarray["shape"][-2:])
    }
    full_coords = dcoords + tuple(lat_lon_dims.keys())
    full_dims = dict(**dims, **lat_lon_dims)

    # all chunk dimensions are 1 except for lat/lon or x/y
    zarray["chunks"] = [
        1 if c not in lat_lon_dims else lat_lon_dims[c] for c in full_coords
    ]
    zarray["shape"] = [full_dims[k] for k in full_coords]
    if zarray["fill_value"] is None:
        # Check dtype first?
        zarray["fill_value"] = np.NaN

    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in full_coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    idata = data.set_index(["time", "step", "level"]).sort_index()

    for idx in itertools.product(*[range(dims[k]) for k in dcoords]):
        # Build an iterator over each of the single dimension chunks
        # TODO Replace this with a reindex operation and iterate the result if the .loc call is slow inside the loop
        dim_idx = {k: v for k, v in zip(dcoords, idx)}

        iloc: tuple[Any, ...] = (
            times[tuple([dim_idx[k] for k in coords["time"]])],
            steps[tuple([dim_idx[k] for k in coords["step"]])],
        )
        if lvals is not None:
            iloc = iloc + (lvals[idx[-1]],)  # type:ignore[assignment]

        try:
            # Squeeze if needed to get a series. Noop if already a series Df has multiple rows
            dval = idata.loc[iloc].squeeze()
        except KeyError:
            logger.info(f"Error getting vals {iloc} for in path {key}")
            continue

        assert isinstance(
            dval, pd.Series
        ), f"Got multiple values for iloc {iloc} in key {key}: {dval}"

        if pd.isna(dval.inline_value):
            # List of [URI(Str), offset(Int), length(Int)] using python (not numpy) types.
            record = [dval.uri, dval.offset.item(), dval.length.item()]
        else:
            record = dval.inline_value
        # lat/lon y/x have only the zero chunk
        vkey = ".".join([str(v) for v in (idx + (0, 0))])
        zstore[f"{key}/{vkey}"] = record


@unique
class AggregationType(Enum):
    """
    ENUM for aggregation types
    TODO is this useful elsewhere?
    """

    HORIZON = "horizon"
    VALID_TIME = "valid_time"
    RUN_TIME = "run_time"
    BEST_AVAILABLE = "best_available"

def reinflate_grib_store(
        axes: list[pd.Index],
        aggregation_type: AggregationType,
        chunk_index: pd.DataFrame,
        zarr_ref_store: dict,
) -> dict:
    """
    Given a zarr_store hierarchy, pull out the variables present in the chunks dataframe and reinflate the zarr
    variables adding any needed dimensions. This is a select operation - based on the time axis provided.
    Assumes everything is stored in hours per grib convention.
    # TODO finish & validate valid_time, run_time & best_available aggregation modes

    :param axes: a list of new axes for aggregation
    :param aggregation_type: the type of fmrc aggregation
    :param chunk_index: a dataframe containing the kerchunk index
    :param zarr_ref_store: the deflated (chunks removed) zarr store
    :return: the inflated zarr store
    """
    # Make a deep copy so we don't modify the input
    zstore = copy.deepcopy(zarr_ref_store["refs"])

    axes_by_name: dict[str, pd.Index] = {pdi.name: pdi for pdi in axes}
    # Validate axis names
    time_dims: dict[str, int] = {}
    time_coords: dict[str, tuple[str, ...]] = {}
    # TODO: add a data class or other method of typing and validating the variables created in this if block
    if aggregation_type == AggregationType.HORIZON:
        # Use index length horizons containing timedelta ranges for the set of steps
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["valid_time"] = len(axes_by_name["valid_time"])

        time_coords["step"] = ("step", "valid_time")
        time_coords["valid_time"] = ("step", "valid_time")
        time_coords["time"] = ("step", "valid_time")
        time_coords["datavar"] = ("step", "valid_time")

        steps = create_steps(axes_by_name["step"], time_dims["valid_time"])
        valid_times = np.tile(
            axes_by_name["valid_time"].to_numpy(), (time_dims["step"], 1)
        )
        times = valid_times - steps

    elif aggregation_type == AggregationType.VALID_TIME:
        # Provide an index of steps and an index of valid times
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["valid_time"] = len(axes_by_name["valid_time"])

        time_coords["step"] = ("step",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time", "step")
        time_coords["datavar"] = ("valid_time", "step")

        steps = axes_by_name["step"].to_numpy()
        valid_times = axes_by_name["valid_time"].to_numpy()

        steps2d = np.tile(axes_by_name["step"], (time_dims["valid_time"], 1))
        valid_times2d = np.tile(
            np.reshape(axes_by_name["valid_time"], (-1, 1)), (1, time_dims["step"])
        )
        times = valid_times2d - steps2d

    elif aggregation_type == AggregationType.RUN_TIME:
        # Provide an index of steps and an index of run times.
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["time"] = len(axes_by_name["time"])

        time_coords["step"] = ("step",)
        time_coords["valid_time"] = ("time", "step")
        time_coords["time"] = ("time",)
        time_coords["datavar"] = ("time", "step")

        steps = axes_by_name["step"].to_numpy()
        times = axes_by_name["time"].to_numpy()

        # The valid times will be runtimes by steps
        steps2d = np.tile(axes_by_name["step"], (time_dims["time"], 1))
        times2d = np.tile(
            np.reshape(axes_by_name["time"], (-1, 1)), (1, time_dims["step"])
        )
        valid_times = times2d + steps2d

    elif aggregation_type == AggregationType.BEST_AVAILABLE:
        time_dims["valid_time"] = len(axes_by_name["valid_time"])
        assert (
                len(axes_by_name["time"]) == 1
        ), "The time axes must describe a single 'as of' date for best available"
        reference_time = axes_by_name["time"].to_numpy()[0]

        time_coords["step"] = ("valid_time",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time",)
        time_coords["datavar"] = ("valid_time",)

        valid_times = axes_by_name["valid_time"].to_numpy()
        times = np.where(valid_times <= reference_time, valid_times, reference_time)
        steps = valid_times - times
    else:
        raise RuntimeError(f"Invalid aggregation_type argument: {aggregation_type}")

    # Copy all the groups that contain variables in the chunk dataset
    unique_groups = chunk_index.set_index(
        ["varname", "stepType", "typeOfLevel"]
    ).index.unique()

    # Drop keys not in the unique groups
    for key in list(zstore.keys()):
        # Separate the key as a path keeping only: varname, stepType and typeOfLevel
        # Treat root keys like ".zgroup" as special and return an empty tuple
        lookup = tuple(
            [val for val in os.path.dirname(key).split("/")[:3] if val != ""]
        )
        if lookup not in unique_groups:
            del zstore[key]

    # Now update the zstore for each variable.
    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        base_path = "/".join(key)
        lvals = group.level.unique()
        dims = time_dims.copy()
        coords = time_coords.copy()
        if len(lvals) == 1:
            lvals = lvals.squeeze()
            dims[key[2]] = 0
        elif len(lvals) > 1:
            lvals = np.sort(lvals)
            # multipel levels
            dims[key[2]] = len(lvals)
            coords["datavar"] += (key[2],)
        else:
            raise ValueError("")

        # Convert to floating point seconds
        # td.astype("timedelta64[s]").astype(float) / 3600  # Convert to floating point hours
        store_coord_var(
            key=f"{base_path}/time",
            zstore=zstore,
            coords=time_coords["time"],
            data=times.astype("datetime64[s]"),
        )

        store_coord_var(
            key=f"{base_path}/valid_time",
            zstore=zstore,
            coords=time_coords["valid_time"],
            data=valid_times.astype("datetime64[s]"),
        )

        store_coord_var(
            key=f"{base_path}/step",
            zstore=zstore,
            coords=time_coords["step"],
            data=steps.astype("timedelta64[s]").astype("float64") / 3600.0,
        )

        store_coord_var(
            key=f"{base_path}/{key[2]}",
            zstore=zstore,
            coords=(key[2],) if lvals.shape else (),
            data=lvals,  # all grib levels are floats
        )

        store_data_var(
            key=f"{base_path}/{key[0]}",
            zstore=zstore,
            dims=dims,
            coords=coords,
            data=group,
            steps=steps,
            times=times,
            lvals=lvals if lvals.shape else None,
        )

    return dict(refs=zstore, version=1)


def read_store(fpath: str) -> dict:
    """
    Cached method for loading the static zarr store from a metadata path
    :param metadata_path: the path (usually gcs) to the metadata directory
    :return: a kerchunk zarr store reference spec dictionary (defalated)
    """
    with fsspec.open(fpath, "rb") as f:
        compressed = f.read()
    logger.info("Read %d bytes from %s", len(compressed), fpath)
    zarr_store = ujson.loads(gzip.decompress(compressed).decode())
    return zarr_store

HORIZONS = {
    "hrrr-conus-sfcf": [np.timedelta64(val, "h") for val in [6, 24, 48]],
    "hrrr-conus-subhf": [np.timedelta64(val, "h") for val in [6, 12, 18]],
    "gfs-atmos-pgrb2-0p25": [np.timedelta64(val, "h") for val in [48, 96, 168, 384]],
}


class CamusProvider(Plugin):
    name = 'xreds_datasets'
    dataset_mapping: dict = {}
    datasets: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        fs = fsspec.filesystem('file')
        with fs.open(settings.datasets_mapping_file, 'r') as f:
            dataset_mapping = json.load(f)

        hrrr_coords = xr.open_zarr(
            "gs://inbox.prod.camus-infra.camus.store/nodd/hrrr/conus/projected_coordinates/v1.0/coords.zarr"
        )
        logger.info("Loaded static hrrr coords")

        tstamp = pd.Timestamp.now()

        for key, dataset_spec in dataset_mapping.items():
            dataset_spec["metadata"] = read_store(dataset_spec["metadata_path"])
            dataset_spec["tstamp"] = tstamp
            if "hrrr" in key:
                dataset_spec["coords"] = hrrr_coords


        dataset_horizon_mapping = {}
        for key, dataset_spec in dataset_mapping.items():
            for horizon in HORIZONS[key]:
                dspec = copy.deepcopy(dataset_spec)
                dspec["horizon"] = horizon
                dataset_horizon_mapping[f"{key}_{str(horizon.astype(int))}-hours"] = dspec


        logger.info("Loaded static dataset metadata for %s", dataset_horizon_mapping.keys())

        self.dataset_mapping = dataset_horizon_mapping

    @hookimpl
    def get_datasets(self):
        return self.dataset_mapping.keys()
    
    @hookimpl
    def get_dataset(self, dataset_id: str) -> xr.Dataset:
        cache_key = f"dataset-{dataset_id}"

        cached_ds = self.datasets.get(cache_key, None)
        if cached_ds:
            if (datetime.datetime.now() - cached_ds['date']).seconds < (10 * 60):
                logger.info(f'Using cached dataset for {dataset_id}')
                return cached_ds['dataset']
            else:
                logger.info(f'Cached dataset for {dataset_id} is stale, reloading...')
                self.datasets.pop(cache_key, None)
        else:
            logger.info(f'No dataset found in cache for {dataset_id}, loading...')


        dataset_spec = self.dataset_mapping[dataset_id]
        dataset_index = dataset_spec["index"]
        dataset_vars = dataset_spec["variables"]
        dataset_metadata = dataset_spec["metadata"]
        dataset_horizon = dataset_spec["horizon"]
        dataset_tstamp = dataset_spec["tstamp"]

        if dataset_id.startswith("hrrr-conus-sfcf"):
            runtime_step = (
                np.timedelta64(6, "h")
                if dataset_horizon > np.timedelta64(18, "h")
                else np.timedelta64(1, "h")
            )

            selected_horizons = pd.timedelta_range(
                start=dataset_horizon - runtime_step,
                end=dataset_horizon,
                freq="60min",
                closed="right",
                name=f"{dataset_horizon.astype(int)} hour"
            )
            start = dataset_tstamp.floor("D") - np.timedelta64(1, "D")
            naive_first_runtime = start - selected_horizons[0]

            runtime_offset = (
                    pd.Timestamp(naive_first_runtime).floor(pd.Timedelta(runtime_step))
                    - naive_first_runtime
            )

            axes = [
                pd.Index(
                    [selected_horizons],
                    name="step"
                ),
                pd.date_range(
                    start=start - runtime_step + runtime_offset,
                    end=dataset_tstamp.floor("h") + dataset_horizon,
                    freq="60min",
                    name="valid_time"
                )
            ]

        elif dataset_id.startswith("hrrr-conus-subhf"):
            axes = [
                pd.Index(
                    [
                        pd.timedelta_range(start="120 minutes", end="180 minutes", freq="15min", closed="right", name="003 hour"),
                    ],
                    name="step"
                ),
                # Must start the valid time range at 15 minutes after!
                pd.date_range("2023-10-28T00:15", "2023-10-30T12:00", freq="15min", name="valid_time")
            ]

        elif dataset_id.startswith("gfs-atmos-pgrb2-0p25"):

            axes = [
                pd.Index(
                    [
                        pd.timedelta_range(start="360 minutes", end="720 minutes", freq="60min", closed="right", name="6-12 hour"),
                    ],
                    name="step"
                ),
                pd.date_range("2023-10-28T01:00", "2023-10-30T00:00", freq="60min", name="valid_time")
            ]

        else:
            raise RuntimeError("Unknown dataset id %s", dataset_id)

        # Get the kerchunk index for the axes given variables and NODD model.
        k_index = get_kerchunk_index(axes, dataset_vars, dataset_index)

        logger.warning("Got %d chunks for dset %s with %s axes", len(k_index), dataset_id, axes)

        # Use the kerchunk index to reinflate the zarr store - inserting the references for the variables
        zstore = reinflate_grib_store(
            axes=axes,
            aggregation_type=AggregationType.HORIZON,
            chunk_index=k_index,
            zarr_ref_store=dataset_metadata,
        )

        # Open the zarr store with fsspec as a datatree
        dtree = datatree.open_datatree(
            fsspec.filesystem("reference", fo=zstore).get_mapper(""),
            engine="zarr",
            consolidated=False,
        )

        ds = dtree[dataset_vars[0]].to_dataset().squeeze()



        for dname in dataset_vars:
            dset = dtree[dname].to_dataset().squeeze()
            split_name = dname.split('/')

            # Drop the first var an add it back with fully qualified name
            if split_name[0] in ds.variables.keys():
                ds = ds.drop(split_name[0])

            ds["_".join(split_name)] = dset[split_name[0]]


        logger.warning("%s: %s", dataset_id, ds)

        # When to add this - does it help?
        #ds = ds.rio.write_crs(4326)

        if "coords" in dataset_spec:
            # HRRR

            # Add the project coordinate variables
            coords_data = dataset_spec["coords"]
            ds["x"] = (("x",), da.from_array(coords_data.x.values, chunks=coords_data.x.shape))
            ds["y"] = (("y",), da.from_array(coords_data.y.values, chunks=coords_data.y.shape))

            # Must drop scalar vertical coordinate till frontend bug is fixed
            ds = ds.drop_vars(["step", "time", "heightAboveGround"])

            # In order, assign the cords, build the index, set the attrs
            # Assign the coordinates
            ds = ds.assign_coords(
                valid_times=ds.valid_time,
                y=ds.y,
                x=ds.x,
            )

            # Set the index
            ds = ds.set_index(valid_times="valid_time")

            # Add the attributes
            ds.x.attrs["axis"] = "X"
            ds.x.attrs["long_name"] = "x coordinate of projection"
            ds.x.attrs["standard_name"] = "projection_x_coordinate"
            ds.x.attrs["unit"] = "m"

            ds.y.attrs["axis"] = "Y"
            ds.y.attrs["long_name"] = "y coordinate of projection"
            ds.y.attrs["standard_name"] = "projection_y_coordinate"
            ds.y.attrs["unit"] = "m"

            ds.valid_times.attrs["axis"] = "T"
            ds.valid_times.attrs['standard_name'] = 'time'

            #ds.longitude.attrs["axis"] = "Y"
            ds.longitude.attrs["long_name"] = "longitude"
            ds.latitude.attrs["standard_name"] = "longitude"
            ds.latitude.attrs["units"] = "degrees_east"

            #ds.latitude.attrs["axis"] = "X"
            ds.latitude.attrs["long_name"] = "latitude"
            ds.latitude.attrs["standard_name"] = "latitude"
            ds.latitude.attrs["units"] = "degrees_north"

        else:
            # Drop vars that break stuff
            #ds = ds.drop_vars(["step", "time", "surface"])
            # Must drop scalar vertical coordinate till frontend bug is fixed
            ds = ds.drop_vars(["step", "time", "heightAboveGround"])

            # In order, assign the cords, build the index, set the attrs
            # Assign coords
            ds = ds.assign_coords(
                valid_times=ds.valid_time,
                latitude=ds.latitude,
                longitude=(((ds.longitude + 180) % 360) - 180),
            )

            # Set index
            ds = ds.set_index(valid_times="valid_time")

            # Assign names
            ds.valid_times.attrs["axis"] = "T"
            ds.valid_times.attrs['standard_name'] = 'time'
            ds.latitude.attrs["axis"] = "Y"
            ds.longitude.attrs["axis"] = "X"

        logger.warning("%s\n%s", dataset_id, ds.cf)

        self.datasets[cache_key] = {
            'dataset': ds,
            'date': datetime.datetime.now()
        }

        if cache_key in self.datasets:
            logger.info(f'Loaded and cached dataset for {dataset_id}')
        else: 
            logger.info(f'Loaded dataset for {dataset_id}. Not cached due to size or current cache score')

        return ds
