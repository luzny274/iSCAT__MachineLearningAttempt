
prefix = "TwoHundredThousand_"

import distutils.dir_util
import os

def create_path(path):
    isExist = os.path.exists(path)    
    if not isExist:      
        os.makedirs(path)

oldClasses = 99
newClasses = 3


mul = int(oldClasses / newClasses)
for i in range(newClasses):
    for it in range(mul):
        num = i * mul + it + 1
        src_dir = prefix + "Sorted/" + f'{num:03d}'
        dest_dir = prefix + "3Classes/" + f'{i:03d}'
        create_path(dest_dir)
        
        print("***")
        print("\nSource folder: " + src_dir)
        print("Dest folder: " + dest_dir)
         
        distutils.dir_util.copy_tree(src_dir, dest_dir)