# smuLymphNet

Deep learning models to detect and quantify immune morphometric features in lymph nodes using multiscale fully convolutional networks. 

## Pipeline summary 

1. Trained germinal centre and sinus segmentation PyTorch models
2. Inference pipeline: generates entire WSI segmentation mask
3. Quantification pipeline:


## 1. Models

The trained pytorch models for germinal centre and sinus segmentation at 10x magnification are under the ./models folder.

1. gc_multi.pth: germinal centre segmentation
2. sinus_multi.pth: sinus segmentation

## 2. Inference

Perform inference at 10x magnification using trained germinal centre and sinus models. Example usage

```python
python ./src/inference.py -wp /folder/with/wsi -sp /output_folder/ -gm /models/gc_multi.pth -sm /models/sinus_multi.pth
```

* `wp`: `str`, the path to the folder containing original WSI (or of a single WSI).
* `sp` : `str`, path to folder to save down segmentation masks.
* `gm`: `str` path of trained pytorch germinal multiscale model
* `sm`: `str` path of trained pytorch sinus multiscale model
* `gt`: `str` threshold for germinal prediction
* `st`: `str` threshold for sinus prediction
* `bl`: `str` base magnification level

## 3. Quantification

Once the segmentation masks have been generated, we can quantify the segmented features. Example usage

```python
python ./src/quantify.py -wp /folder/with/wsi -mp /folder/with/segmentation_masks -sp /folder/to_save_output
```

* `wp`: `str`, the path to the folder containing original WSI (or of a single WSI).
* `mp` : `str`, the path to the folder containng the segmentation masks.
* `sp`: `str` the path to save outputs


## Credits

The pipeline was written by the [Cancer Bioinformatics][url_cb] group at [King's College London][url_kcl], UK.

Development and implementation by [Gregory Verghese](gregory.e.verghese@kcl.ac.uk), [Mengyuan Li](mengyuan.3.li@@kcl.ac.uk), [Nikhil Cherian](nikhilcherian30@gmail.com). 

Study concept and design [Gregory Verghese](gregory.verghese.@kcl.ac.uk), [Mengyuan Li](mengyuan.3.li@@kcl.ac.uk) and [Anita Grigoriadis](anita.grigoriadis@kcl.ac.uk).

[url_cb]: http://cancerbioinformatics.co.uk/
[url_kcl]: https://www.kcl.ac.uk/

