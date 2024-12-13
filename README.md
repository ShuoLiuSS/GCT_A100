# Paper Info.
Hyunseung Kim, Jonggeol Na*, and Won Bo Lee*, "Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention, " J. Chem. Inf. Model. 2021, 61, 12, 5804-5814  
[Featured as JCIM journal supplentary cover]  https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01289  

**Generative Chemical Transformer (GCT) directly designs hit-like materials matched given target properties. Transformer's high level of context-awareness ability of chemical language aids in designing molecules more realistic and vaild.**

# Citation
```
@article{kim2021generative,
  title={Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention},
  author={Kim, Hyunseung and Na, Jonggeol and Lee, Won Bo},
  journal={Journal of Chemical Information and Modeling},
  volume={61},
  number={12},
  pages={5804--5814},
  year={2021},
  publisher={ACS Publications}
}

```
# Implementation
This code was tested in Windows OS
1. Set up your anaconda environment with the following code:
```
conda env create -f molgct_env.yaml
```

2. Run to train GCT:
```
python train.py

On Google Colab: !/content/drive/MyDrive/molgct_env/miniconda3/envs/molgct/bin/python /content/drive/MyDrive/GCT/train.py 
```
3. Continue to train GCT from previous trained weights:
```
python train.py -load_weights w_trained -epochs 2
```

4. Run to infer molecules with the trained GCT:
```
python inference.py
```


# References
Original GCT code was borrowed and modified from:https://github.com/Hyunseung-Kim/molGCT  
Basic Transformer code was borrowed and modified from: https://github.com/SamLynnEvans/Transformer  
Molecular data were borrowed from: https://github.com/molecularsets/moses/tree/master/data
