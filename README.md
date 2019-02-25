# Handw_recognition
This repository contains the code of our submission for [Accenture Digital Challenge.](https://www.hackerearth.com/challenges/hackathon/accenture-imagesing/). Other contributor is [Ashutosh Sancheti](https://www.linkedin.com/in/ashutosh-sancheti-852b77177/?lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_top%3BVOKLProtS%2B%2BxjPnIAH%2BHDA%3D%3D&licu=urn%3Ali%3Acontrol%3Ad_flagship3_search_srp_top-search_srp_result&lici=uUoyQDHLSLCG2Y1bUsnGrg%3D%3D).

## Introduction
The participants were asked to use their skills with image recognition to find an innovative solution based on artificial intelligence able to recognize hand written texts and manuscripts (at least in block letters) in latin alphabet and letters.

## Demo Video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/QrRokLO14is/0.jpg)](http://www.youtube.com/watch?v=QrRokLO14is "Handwritten Text Recognition")

## Dataset:
* To use test_iam_dataset.ipynb in `src/utils/`, create credentials.json using credentials.json.example and editing the appropriate field. The username and password can be obtained from http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php.

## Steps to Reproduce
1. Clone the repository to your machine.
```bash 
git clone https://github.com/limosin/Handw_recognition`
```
2. Setup the virtual environemnt using conda. 
```bash
conda env create -f environment.yml
```
3. Download the pretrained weights from this [link](https://drive.google.com/open?id=1xxsuQoYfZbG4nJfRMznl7jdcWIvRAWOR).
4. Extract to '/models' in the main directory(Make this folder if it does not exists).
5. Now for performing OCR on a 'image_example.jpg', open a terminal in the main directory and enter this command.
```bash
python OCR.py -f <image_example>
```
__You can refer to the Demo for more detailed steps.__

### SClite installation
1) Navigate to src/utils/ and find `sctk-2.4.10-20151007-1312Z.tar.bz2`. 
3) Untar sctk-2.4.10
4) Install sctk-2.4.10 by following sctk-2.4.10/INSTALL
5) Check sctk-2.4.10/bin contains built programs

## References
The original prototype was built by [Thomas Delteil](https://github.com/ThomasDelteil).
