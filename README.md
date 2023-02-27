# smuLymphNet

Deep learning models to detect and quantify immune morphometric features in lymph nodes using multiscale fully convolutional networks. 

## Pipeline summary 

1. Trained germinal centre and sinus segmentation PyTorch models
2. Inference pipeline: generates entire WSI segmentation mask
3. Quantification pipeline:

## Quantification

Once the segmentation masks have been generated, we can quantify the segmented features. Example usage

```python
python ./src/quantify.py -wp /folder/with/wsi -mp /folder/with/segmentation_masks -sp /folder/to_save_output
```

## Credits

The pipeline was written by the [Cancer Bioinformatics][url_cb] group at [King's College London][url_kcl], UK.

Development and implementation by [Gregory Verghese](gregory.e.verghese@kcl.ac.uk), [Mengyuan Li](mengyuan.3.li@@kcl.ac.uk), [Nikhil Cherian](nikhilcherian30@gmail.com). 

Study concept and design [Gregory Verghese](gregory.verghese.@kcl.ac.uk), [Mengyuan Li](mengyuan.3.li@@kcl.ac.uk) and [Anita Grigoriadis](anita.grigoriadis@kcl.ac.uk).

[url_cb]: http://cancerbioinformatics.co.uk/
[url_kcl]: https://www.kcl.ac.uk/

