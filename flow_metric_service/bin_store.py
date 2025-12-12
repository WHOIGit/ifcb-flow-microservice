"""IFCB data store for accessing ADC data."""

from ifcb import DataDirectory
from storage.object import ObjectStore
from storage.utils import ReadonlyStore
from storage.config_builder import register_store


class BaseIFCBStore(ObjectStore):
    """Base class for IFCB stores."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def put(self, key, value):
        raise NotImplementedError("IFCB stores are read-only")

    def delete(self, key):
        raise NotImplementedError("IFCB stores are read-only")

    def exists(self, key):
        try:
            data_dir = DataDirectory(self.data_dir)
            return data_dir.has_key(key)
        except Exception:
            return False


@register_store
class IFCBADCStore(BaseIFCBStore):
    """Store for accessing IFCB ADC data."""

    def get(self, key):
        data_dir = DataDirectory(self.data_dir)
        ifcb_bin = data_dir[key]

        # Create column name mapping from schema
        adc = ifcb_bin.adc
        schema = ifcb_bin.schema

        # Build reverse mapping: column index -> name
        column_mapping = {}
        for attr in dir(schema):
            if not attr.startswith('_') and not callable(getattr(schema, attr)):
                val = getattr(schema, attr)
                if isinstance(val, int):
                    column_mapping[val] = attr

        # Rename columns
        adc = adc.rename(columns=column_mapping)
        return adc
