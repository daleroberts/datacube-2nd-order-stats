import logging
import numpy as np
import xarray as xr

from datacube.model import Measurement
from datacube_stats.statistics import Statistic
from copy import copy

from .fast import smad, emad, bcmad, geomedian

LOG = logging.getLogger(__name__)


def sizefmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


class CosineDistanceMAD(Statistic):

    def __init__(self, num_threads=3):
        super().__init__()
        self.num_threads = num_threads
        LOG.info("num_threads: %i", num_threads)

    def compute(self, data: xr.Dataset) -> xr.Dataset:
        squashed = data.to_array().transpose("y", "x", "time", "variable")

        fdata = squashed.data.astype(np.float32) / 10000.
        fdata[(squashed.data == -999)] = np.nan
        fdata[(squashed.data == 0)] = np.nan

        del squashed

        LOG.info("Data array size: %s", sizefmt(fdata.nbytes))

        mask = np.isnan(fdata).any(axis=2)
        ndepth = np.count_nonzero(mask, axis=-1)
        mindepth, mediandepth, maxdepth = np.min(ndepth), np.median(ndepth), np.max(ndepth)
        LOG.info("data mindepth: %s maxdepth: %s mediandepth: %s", mindepth, maxdepth, mediandepth)

        LOG.info("Computing geometric median mosaic")

        gm = geomedian(fdata, num_threads=self.num_threads)

        LOG.info("Computing spectral MAD mosaic")

        dev = smad(fdata, gm, num_threads=self.num_threads)

        da = xr.DataArray(dev, dims=("y", "x"), name="dev")
        return xr.Dataset(data_vars={"dev": da})

    def measurements(self, m):
        mm = [Measurement(name="dev", dtype="float32", nodata=0, units="1")]
        LOG.debug("Returning measurements: %s", mm)
        return mm


class EuclideanDistanceMAD(Statistic):

    def __init__(self, num_threads=3):
        super().__init__()
        self.num_threads = num_threads
        LOG.info("num_threads: %i", num_threads)

    def compute(self, data: xr.Dataset) -> xr.Dataset:
        squashed = data.to_array().transpose("y", "x", "time", "variable")

        fdata = squashed.data.astype(np.float32) / 10000.
        fdata[(squashed.data == -999)] = np.nan
        fdata[(squashed.data == 0)] = np.nan

        del squashed

        LOG.info("Data array size: %s", sizefmt(fdata.nbytes))

        mask = np.isnan(fdata).any(axis=2)
        ndepth = np.count_nonzero(mask, axis=-1)
        mindepth, mediandepth, maxdepth = np.min(ndepth), np.median(ndepth), np.max(ndepth)
        LOG.info("data mindepth: %s maxdepth: %s mediandepth: %s", mindepth, maxdepth, mediandepth)

        LOG.info("Computing geometric median mosaic")

        gm = geomedian(fdata, num_threads=self.num_threads)

        LOG.info("Computing spectral MAD mosaic")

        dev = emad(fdata, gm, num_threads=self.num_threads)

        da = xr.DataArray(dev, dims=("y", "x"), name="dev")
        return xr.Dataset(data_vars={"dev": da})

    def measurements(self, m):
        mm = [Measurement(name="dev", dtype="float32", nodata=0, units="1")]
        LOG.debug("Returning measurements: %s", mm)
        return mm


class BrayCurtisDistanceMAD(Statistic):

    def __init__(self, num_threads=3):
        super().__init__()
        self.num_threads = num_threads
        LOG.info("num_threads: %i", num_threads)

    def compute(self, data: xr.Dataset) -> xr.Dataset:
        squashed = data.to_array().transpose("y", "x", "time", "variable")

        fdata = squashed.data.astype(np.float32) / 10000.
        fdata[(squashed.data == -999)] = np.nan
        fdata[(squashed.data == 0)] = np.nan

        del squashed

        LOG.info("Data array size: %s", sizefmt(fdata.nbytes))

        mask = np.isnan(fdata).any(axis=2)
        ndepth = np.count_nonzero(mask, axis=-1)
        mindepth, mediandepth, maxdepth = np.min(ndepth), np.median(ndepth), np.max(ndepth)
        LOG.info("data mindepth: %s maxdepth: %s mediandepth: %s", mindepth, maxdepth, mediandepth)

        LOG.info("Computing geometric median mosaic")

        gm = geomedian(fdata, num_threads=self.num_threads)

        LOG.info("Computing Bray Curtis distance MAD mosaic")

        dev = bcmad(fdata, gm, num_threads=self.num_threads)

        da = xr.DataArray(dev, dims=("y", "x"), name="dev")
        return xr.Dataset(data_vars={"dev": da})

    def measurements(self, m):
        mm = [Measurement(name="dev", dtype="float32", nodata=0, units="1")]
        LOG.debug("Returning measurements: %s", mm)
        return mm


class TernaryMAD(Statistic):
    def __init__(self, num_threads=3):
        super().__init__()
        self.num_threads = num_threads
        LOG.info("num_threads: %i", num_threads)

    def compute_on_array(self, data: np.array) -> np.array:
        np.seterr(all="ignore")

        LOG.info("Data array size: %s dimensions: %s", sizefmt(data.nbytes), data.shape)

        mask = np.isnan(data).any(axis=2)
        ndepth = np.count_nonzero(mask, axis=-1)
        mindepth, mediandepth, maxdepth = np.min(ndepth), np.median(ndepth), np.max(ndepth)
        LOG.info("data mindepth: %s maxdepth: %s mediandepth: %s", mindepth, maxdepth, mediandepth)

        LOG.info("Computing geometric median mosaic")

        gm = geomedian(data, num_threads=self.num_threads)

        LOG.info("Computing cosine distance MAD mosaic")

        sdev = smad(data, gm, num_threads=self.num_threads)

        LOG.info("Computing Euclidean distance MAD mosaic")

        edev = emad(data, gm, num_threads=self.num_threads)

        LOG.info("Computing Bray-Curtis distance MAD mosaic")

        bcdev = bcmad(data, gm, num_threads=self.num_threads)

        LOG.info("Stacking results")

        result = np.dstack([sdev, edev, bcdev])

        LOG.info("Mosaic size: %s dimensions: %s", sizefmt(result.nbytes), result.shape)

        return result

    def compute(self, data: xr.Dataset) -> xr.Dataset:
        np.seterr(all="ignore")

        squashed_together_dimensions, normal_datacube_dimensions = self._vars_to_transpose(data)
        squashed = data.to_array(dim="variable").transpose(*squashed_together_dimensions)
        assert squashed.dims == squashed_together_dimensions

        output_coords = copy(squashed.coords)
        if "time" in output_coords:
            del output_coords["time"]
        if "source" in output_coords:
            del output_coords["source"]

        fdata = squashed.data.astype(np.float32) / 10000.
        fdata[(squashed.data == -999)] = np.nan
        fdata[(squashed.data == 0)] = np.nan

        tmp = self.compute_on_array(fdata)

        da = xr.DataArray(tmp[:, :, 0], dims=("y", "x"), name="sdev")
        db = xr.DataArray(tmp[:, :, 1], dims=("y", "x"), name="edev")
        dc = xr.DataArray(tmp[:, :, 2], dims=("y", "x"), name="bcdev")
        ds = xr.Dataset(data_vars={"sdev": da, "edev": db, "bcdev": dc})

        LOG.info("Finished computing")
        return ds

    def measurements(self, m):
        mm = [
            Measurement(name="sdev", dtype="float32", nodata=np.nan, units="1"),
            Measurement(name="edev", dtype="float32", nodata=np.nan, units="1"),
            Measurement(name="bcdev", dtype="float32", nodata=np.nan, units="1"),
        ]
        LOG.debug("Returning measurements: %s", mm)
        return mm

    @staticmethod
    def _vars_to_transpose(data):
        is_projected = "x" in data.dims and "y" in data.dims
        is_geographic = "longitude" in data.dims and "latitude" in data.dims
        if is_projected and is_geographic:
            raise StatsProcessingError("Data to process contains BOTH geographic and projected dimensions")

        elif not is_projected and not is_geographic:
            raise StatsProcessingError("Data to process contains NEITHER geographic nor projected dimensions")

        elif is_projected:
            return ("y", "x", "variable", "time"), ("variable", "y", "x")

        else:
            return ("latitude", "longitude", "variable", "time"), ("variable", "latitude", "longitude")




SMAD = CosineDistanceMAD
EMAD = EuclideanDistanceMAD
BCMAD = BrayCurtisDistanceMAD
