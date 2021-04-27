import tkinter as tk
import tkinter.filedialog as fd
from osgeo import gdal,ogr
import numpy as np
import tkinter.messagebox as tmsg
import os
from sklearn.model_selection import train_test_split
from patchify import patchify, unpatchify
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



root=tk.Tk()
current_dir=os.getcwd()

def readraster(file):
    dataSource = gdal.Open(file)
    band = dataSource.ReadAsArray()
    #print("band")
    #print(band)
    #print("datasource")
    #print(dataSource)
    return(dataSource, band)

def load_image():
    im=fd.askopenfilename(title="open image",initialdir=current_dir[:3],filetype=(("tif file","*tif"),("tiff file","*tiff")))
    st.set(im)
    en.update()
    main_image=im.split("/")[-1]
    #global raster_im
    #raster_im=gdal.Open(main_image)
    #print(raster_im)
    global arr_lc1

    ds_lc1, arr_lc1 = readraster(main_image)
    print(arr_lc1)
    pass
def open_image():
    show_im=raster_im.GetRasterBand(1)
    arr=show_im.ReadAsArray()
    show_im2=plt.imshow(arr)
    plt.show()
    pass

def preprocessing():
    image = arr_lc1
    image = np.reshape(image, (image.shape[1],image.shape[2], image.shape[0]))

    newX = np.reshape(image, (-1, image.shape[2])) 
    #print(newX.shape)

    scaler = StandardScaler().fit(newX)  
    newX = scaler.transform(newX) 
    
    newX = np.reshape(newX, (image.shape[0],image.shape[1],image.shape[2]))

    image_patches = patchify(newX, (3,3,4), step=1) 

    image_patches = np.reshape(image_patches, (image_patches.shape[0]*image_patches.shape[1]* image_patches.shape[2],image_patches.shape[3],image_patches.shape[4],image_patches.shape[5]))

    train_center_pixel= []
    x = 0
    for i in range(757435):
        train_center_pixel.append(image_patches[x])
        x= x+1

    train_center_pixel = np.array(train_center_pixel)
    global input_shape
    input_shape = image_patches[0].shape

    image_patches = np.reshape(image_patches, (image_patches.shape[0],image_patches.shape[3],image_patches.shape[2],image_patches.shape[1]))
    print(image_patches[0].shape)
    #testRatio=0.05
    #def splitTrainTestSet(X, y, testRatio=0.20):
     #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345,stratify=y)
      #  return X_train, X_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(image_patches,train_center_pixel)
    x_train=np.asarray(x_train)
    x_test=np.asarray(x_test)
    y_train=np.asarray(y_train)
    y_test=np.asarray(y_test)
    #np.save("x_train.npy",x_train)
    #np.save("x_test.npy",x_test)
    #np.save("y_train.npy",y_train)
    #np.save("y_test.npy",y_test)
    def saveProcessedData(x_train,x_test,y_train,y_test):
        with open("x_train.npy","bw") as outfile:
            np.save(outfile,x_train)
        with open("x_test.npy","bw") as outfile:
            np.save(outfile,x_test)
        with open("y_train.npy","bw") as outfile:
            np.save(outfile,y_train)
        with open("y_test.npy","bw") as outfile:
            np.save(outfile,y_test)
    saveProcessedData(x_train,x_test,y_train,y_test)
    x_train1=np.load(current_dir+"\\x_train.npy")
    x_test1=np.load(current_dir+"\\x_test.npy")
    y_train1=np.load(current_dir+"\\y_train.npy")
    y_test1=np.load(current_dir+"\\y_test.npy")
    
    print("x_train",x_train1)
    print("x_test",x_test1)
    print("y_train",y_train1)
    print("y_test",y_test1)
    #print (image_patches)
    pass

def classification():
    print("Successfully modeled")
    pass

root.geometry("400x200")
root.title("Analyzing the crop area using Remote sensing")
root.config(bg="gray")
st=tk.StringVar()
st.set("")
fr=tk.Frame(root,bg="gray")
fr.pack(padx=4,pady=4,side="top",fill="both")
en=tk.Entry(fr,textvariable=st,width=40)
en.pack(side="left",padx=5)
bu1=tk.Button(fr,text="Load Image",command=load_image)
bu1.pack(padx=5)
bu2=tk.Button(root,text="Open Image",command=open_image)
bu2.pack(padx=8,side="top",anchor="w")
bu3=tk.Button(root,text="Pre-processing",command=preprocessing)
bu3.pack(padx=8,side="top",anchor="w")
bu4=tk.Button(root,text="Train model",command=classification)
bu4.pack(padx=8,side="top",anchor="w")
root.mainloop()