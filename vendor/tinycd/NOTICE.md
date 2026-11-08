# TinyCD

ASTRA uses the TinyCD change-detection model by Andrea Codegoni, Gabriele Lombardi and Alessandro Ferrari ([arXiv:2207.13159](https://arxiv.org/abs/2207.13159), code at <https://github.com/AndreaCodegoni/Tiny_model_4_CD>).

The authors release the TinyCD code for non-commercial and research purposes only, and the repository has no open-source license. The code is therefore not included here. Run

    python scripts/install_tinycd.py

to download the three model files from upstream commit `1696827ffb7473c448591aaaa4434d4d4934a1ba` into `vendor/tinycd/models/` (ignored by Git). The script checks each file's SHA-256. The files ASTRA was developed with are identical to that commit apart from line endings.

The pretrained LEVIR-CD checkpoint is downloaded separately by `scripts/download_tinycd_weights.py`, comes from the same upstream repository and is under the same terms.
