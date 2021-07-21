""" Facies batch. """
from ..crop_batch import SeismicCropBatch
from ...batchflow import action, inbatch_parallel



class FaciesBatch(SeismicCropBatch):
    """ Batch with facies-specific data loading. """

    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def load_attribute(self, ix, dst, src='amplitudes', src_labels='labels', res_ndim=3, **kwargs):
        """ Load attribute for label at given location.

        Parameters
        ----------
        src : str
            A keyword from :attr:`~Horizon.ATTRIBUTE_TO_METHOD` keys, defining label attribute to make crops from.
        src_labels : str
            Dataset attribute with labels dict.
        locations : str
            Component of batch with locations of crops to load.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`~Horizon.ATTRIBUTE_TO_METHOD` depending on `attribute`
            - or attribute-transforming method :meth:`~Horizon.transform_where_present`.

        Notes
        -----
        This method loads rectified data, e.g. amplitudes are croped relative
        to horizon and will form a straight plane in the resulting crop.
        """
        location = self.get(ix, 'locations')
        label_index = self.get(ix, 'generated')[1]
        label = self.get(ix, src_labels)[label_index]

        label_name = self.get(ix, 'label_names')
        if label.short_name != label_name:
            msg = f"Name `{label.short_name}` of the label loaded by index {label_index} "\
                  f"from {src_labels} does not match label name {label_name} from batch."\
                  f"This might have happened due to items order change in {src_labels} "\
                  f"in between sampler creation and `make_locations` call."
            raise ValueError(msg)

        return label.load_attribute(src=src, location=location, **kwargs)
