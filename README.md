# Cassava Leaf Disease Assistant

Berkin Ozturk | Deep Learning with PyTorch, capstone project

A ResNet18 trained to sort cassava leaf photos into five categories, wired to Claude so the
output comes back as advice a farmer can act on instead of a class index. Wrapped in Gradio.

## The problem

Cassava feeds a lot of people in sub-Saharan Africa, and four diseases account for most of the
crop loss. Spotting them early matters, but expert diagnosis is scarce and the visual
differences are subtle. The photos are taken on phones in the field, so they are messy: bad
light, odd angles, cluttered backgrounds. That is what makes this a harder and more interesting
problem than the clean lab-photo plant datasets.

## Data

[iCassava 2019](https://www.tensorflow.org/datasets/catalog/cassava) (Mwebaze et al.), collected
by Makerere University's AI lab and NaCRRI, labelled by plant pathologists.

- 9,430 labelled images, 5 classes
- Fixed split shipped with the dataset: 5,656 train / 1,889 validation / 1,885 test
- Classes: `cbb` bacterial blight, `cbsd` brown streak disease, `cgm` green mite damage,
  `cmd` mosaic disease, `healthy`
- Heavily imbalanced. CMD has roughly ten times the images of healthy.
- Public download, no Kaggle account needed:
  `https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip`

## Approach

| Piece | Choice | Why |
|---|---|---|
| Baseline | SmallCNN, 4 conv blocks, trained from scratch | Shows what the data alone gives you |
| Main model | ResNet18, ImageNet weights, new 5-way head | Under 6k training images is too few to learn low-level filters from nothing |
| Imbalance | Inverse-frequency class weights in the loss | Stops the rare classes from being ignored |
| Augmentation | Random resized crop, flips, rotation, colour jitter, random erasing | Matches the variation in real field photos |
| Optimizer | AdamW with OneCycle schedule, mixed precision | Fast enough for a free Colab session |
| Early stopping | On validation macro F1, patience 5 | Accuracy hides the rare-class problem here, macro F1 does not |
| Tuning | 5-point sweep over learning rate, batch size, weight decay | Ranks settings in short 6-epoch runs, then the winner is retrained fully |

## Results

Both models on the held-out test set, same split, same augmentation, same loss.

<!-- results:start -->
| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| SmallCNN, from scratch | 74.7% | 0.660 | 0.737 | 0.668 |
| ResNet18, fine-tuned | 86.5% | 0.793 | 0.848 | 0.816 |

Transfer learning is worth +11.8 points of test accuracy on this dataset.
<!-- results:end -->

Per-class precision, recall, and F1, plus the confusion matrices for both models, are in the
notebook under sections 6 and 7.

## The LLM step

The classifier's top three probabilities go into a single Anthropic API call, which returns a
short recommendation written for a farmer. The probabilities are in the prompt on purpose, so
the response can hedge when the classifier is unsure rather than sounding certain about a
coin-flip prediction.

## Running it

Training: open `cassava_capstone.ipynb` in Colab, set the runtime to a T4 GPU, add your
`ANTHROPIC_API_KEY` under Colab Secrets, and run all cells. The Gradio interface launches from
the notebook with a public share link.

Locally, after downloading `cassava_resnet18.pt` from the notebook:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

## What this does not prove

The model is trained and tested on images from one collection effort in Uganda, so the test set
shares its cameras, seasons, and field conditions with the training set. A good test number here
is not evidence that it works on a different farm, phone, or time of year. Nothing in this
project measures that.

The generated advice is also unchecked. It reads well, but no agronomist reviewed it, and the
system will produce fluent recommendations even when the underlying prediction is wrong. That
failure mode is worse than a bare wrong label, because the text sounds authoritative. Real use
would need a field trial and expert review of the LLM output.

## Files

```
cassava_capstone.ipynb   training, tuning, evaluation, LLM call, UI
app.py                   standalone Gradio app
requirements.txt
results.json             metrics, written by the notebook
cassava_resnet18.pt      checkpoint, written by the notebook
```

The notebook runs end to end on a free Colab T4 in roughly two hours, including the dataset
download. It writes `results.json` and the saved checkpoint, which is what `app.py` loads.

## Reference

Mwebaze, E., Gebru, T., Frome, A., Nsumba, S., Tusubira, J. (2019).
*iCassava 2019 Fine-Grained Visual Categorization Challenge.* arXiv:1908.02900.
