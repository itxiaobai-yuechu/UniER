# I. File Replacement Instructions
## 1. wandb_predict.py
Replace the `wandb_predict.py` file in your pykt project with this one. The specific file directory for replacement is as follows:

There is a `wandb_predict.py` in the `example` directory, please replace it directly.

## 2. evaluate_model.py
This file needs to replace the `evaluate_model.py` file in the pykt library. The specific file directory is as follows:

**[Note]**: The file path at point ?in the above figure should be the location where you installed the pykt library in the corresponding conda environment, which is: `(your own conda directory)/anaconda3/envs/(your own environment)/lib/python3.11/site-packages/pykt/models/`.

## 3. train_model.py
This file needs to replace the `train_model.py` file in the pykt library. The specific file directory is as follows:

**[Note]**: The file path at point ?in the above figure should be the location where you installed the pykt library in the corresponding conda environment (which is actually the same directory as the 2nd file `evaluate_model.py`), which is: `(your own conda directory)/anaconda3/envs/(your own environment)/lib/python3.11/site-packages/pykt/models/`.

**[Special Note]**: Compared to the original pykt library file before replacement, this `train_model.py` only adds the code on line 39 as shown in the figure below.

# II. How to Obtain pkm and pkc
## 1. Meaning
?**pkm**: The mastery level of each student in the test set (each row in the test set `test_sequences.csv` represents a student, unrelated to the `uid` in the file) for each knowledge concept. The `pkm` you extract will be a 2D tensor of size (length of test set  number of knowledge concepts). The `MLKC` in the subsequent KG4Ex has the same meaning as the `pkm` here, and is also extracted from here.

?**pkc**: The probability of each knowledge concept appearing at the next time step for each student in the test set (each row in `test_sequences.csv` represents a student, unrelated to the `uid` in the file). The `pkc` you extract will be a 2D tensor of size (length of test set  number of knowledge concepts). The `PKC` in the subsequent KG4Ex has the same meaning as the `pkc` here, and is also extracted from here.

## 2. Acquisition
### ?Obtaining pkm
A. First, replace the above files, then train dkt normally, which means going from data preprocessing all the way to finishing the model training, and seeing the `saved_model` folder save the dkt model.
B. Open `wandb_predict.py` in your pykt project, then modify the following place:

Change the file name in 2 places to `pkm.pth`
C. The `pkm` file will appear in the folder of the pre-trained dkt you selected within the `saved_model` directory. As shown below:

### ?Obtaining pkc
A. First, replace the above files, then open the `train_model.py` file in the pykt library, and modify the line 40 code mentioned above:

B. Then train dkt, which means going from data preprocessing all the way to finishing the model training, and seeing the `saved_model` folder save the dkt model.
C. Open `wandb_predict.py` in your pykt project, then modify the following place:

Change the file name in 2 places to `pkc.pth`
D. The `pkc` file will appear in the folder of the pre-trained dkt you selected within the `saved_model` directory. As shown below:

E. **Attention!!** After obtaining the `pkc` file, it is essential to restore line 40 in the `train_model.py` file from `t_ones.double()` back to the original `t.double()` to avoid errors the next time you use dkt.


