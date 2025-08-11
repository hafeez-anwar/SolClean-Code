{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b8efb41-b7da-4831-9e20-a61b34e6cc20",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python\n",
    "# coding: utf-8\n",
    "\n",
    "import sys\n",
    "import yaml\n",
    "import argparse\n",
    "\n",
    "from encode import encode_images\n",
    "from classify import train_and_evaluate_svm\n",
    "\n",
    "\n",
    "def main():\n",
    "    parser = argparse.ArgumentParser(description=\"Encode images with timm backbones and classify with linear SVM.\")\n",
    "    parser.add_argument(\"--config\", required=True, help=\"Path to config.yaml\")\n",
    "    parser.add_argument(\"--encode-only\", action=\"store_true\", help=\"Run encoding only\")\n",
    "    parser.add_argument(\"--classify-only\", action=\"store_true\", help=\"Run classification only\")\n",
    "    args = parser.parse_args()\n",
    "\n",
    "    with open(args.config, \"r\") as f:\n",
    "        cfg = yaml.safe_load(f)\n",
    "\n",
    "    if not args.classify_only:\n",
    "        encode_images(cfg)\n",
    "\n",
    "    if not args.encode_only:\n",
    "        train_and_evaluate_svm(cfg)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
