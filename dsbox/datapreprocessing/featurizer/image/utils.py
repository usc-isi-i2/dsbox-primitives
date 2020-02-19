import logging
import collections
from d3m.metadata.base import ALL_ELEMENTS
from d3m import container

CONTAINER_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json'
logger = logging.getLogger(__name__)

class image_utils:
    @staticmethod
    def get_image_path(input_df) -> str:
        """
            function used to get the abs path of input images
        """
        target_index = []
        location_base_uris = []
        elements_amount = input_df.metadata.query((ALL_ELEMENTS,))['dimension']['length']
        for selector_index in range(elements_amount):
            each_selector = input_df.metadata.query((ALL_ELEMENTS, selector_index))
            mime_types_found = False
            # here we assume only one column shows the location of the target attribute
            #print(each_selector)
            if 'mime_types' in each_selector:
                mime_types = (each_selector['mime_types'])
                mime_types_found = True
            elif 'media_types' in each_selector:
                mime_types = (each_selector['media_types'])
                mime_types_found = True
            # do following step only when mime type attriubte found
            if mime_types_found:
                for each_type in mime_types:
                    if ('image' in each_type):
                        target_index.append(selector_index)
                        location_base_uris.append(each_selector['location_base_uris'][0])
                        # column_name = each_selector['name']
                        break
        # if no 'image' related mime_types found, return a ndarray with each dimension's length equal to 0
        if (len(target_index) == 0):
            # raise exceptions.InvalidArgumentValueError("no image related metadata found!")
            logger.error("[ERROR] No image related column found!")
        elif len(target_index) > 1:
            logger.info("[INFO] Multiple image columns found in the input, this primitive can only handle one column.")
        location_base_uris = location_base_uris[0]
        if location_base_uris[0:7] == 'file://':
            location_base_uris = location_base_uris[7:]
        return location_base_uris

    @staticmethod
    def generate_all_metadata(output_dataFrame) -> container.DataFrame:
        """
            function that used to add metadata for all keras primitives outputs
        """
        # reset column names to be strings
        column_names_replaced = {}
        for each_col_name in output_dataFrame.columns:
            if isinstance(each_col_name, int):
                column_names_replaced[each_col_name] = "col_{}".format(str(each_col_name))
        output_dataFrame.rename(columns=column_names_replaced, inplace=True)
        # add d3mIndex metadata
        index_metadata_selector = (ALL_ELEMENTS, 0)
        index_metadata = {
            'name': 'd3mIndex',
            'semantic_types': (
                "http://schema.org/Integer",
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey'), 
            'structural_type': int 
            }
        output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=index_metadata, selector=index_metadata_selector)
        # update 2019.5.15: update dataframe's shape metadatapart!
        metadata_shape_part_dict = image_utils.generate_metadata_shape_part(value=output_dataFrame, selector=())
        for each_selector, each_metadata in metadata_shape_part_dict.items():
            output_dataFrame.metadata = output_dataFrame.metadata.update(selector=each_selector, metadata=each_metadata)

        for each_column in range(1, output_dataFrame.shape[1]):
            metadata_selector = (ALL_ELEMENTS, each_column)
            metadata_each_column = {
            'name': output_dataFrame.columns[each_column],
            'semantic_types': (
                'https://metadata.datadrivendiscovery.org/types/TabularColumn', 
                'https://metadata.datadrivendiscovery.org/types/Attribute'), 
            'structural_type': float
            }
            output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=metadata_each_column, selector=metadata_selector)
        return output_dataFrame

    @staticmethod
    def generate_metadata_shape_part(value, selector) -> dict:
        """
        recursively generate all metadata for shape part, return a dict
        :param value:
        :param selector:
        :return:
        """
        generated_metadata: dict = {}
        generated_metadata['schema'] = CONTAINER_SCHEMA_VERSION
        if isinstance(value, container.Dataset):  # type: ignore
            generated_metadata['dimension'] = {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': len(value),
            }

            metadata_dict = collections.OrderedDict([(selector, generated_metadata)])

            for k, v in value.items():
                metadata_dict.update(generate_metadata_shape_part(v, selector + (k,)))

            # It is unlikely that metadata is equal across dataset resources, so we do not try to compact metadata here.

            return metadata_dict

        if isinstance(value, container.DataFrame):  # type: ignore
            generated_metadata['semantic_types'] = ['https://metadata.datadrivendiscovery.org/types/Table']

            generated_metadata['dimension'] = {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': value.shape[0],
            }

            metadata_dict = collections.OrderedDict([(selector, generated_metadata)])

            # Reusing the variable for next dimension.
            generated_metadata = {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': value.shape[1],
                },
            }

            selector_all_rows = selector + (ALL_ELEMENTS,)
            metadata_dict[selector_all_rows] = generated_metadata
            return metadata_dict
