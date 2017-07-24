import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgbToG(img):
 npImg=np.asarray(img)
 r=0.2125
 g=0.7154
 b=0.0721
 gsImg=r*npImg[:,:,0]+g*npImg[:,:,1]+b*npImg[:,:,2]
 return gsImg
def imgToGray(imgF):
 imageP=Image.open(imgF)
 imnp=np.asarray(imageP)
 imageP.close() #Close opened image
 ims=len(imnp.shape)
 if ims == 3:
  imgG=rgbToG(imnp)
 elif ims ==2:
  imgG=imnp
 return imgG
def borderFilter(np_img):
 np_imgx=np_img.copy().astype('float')
 npimgs=np_imgx.shape
 imgx=npimgs[0]
 imgy=npimgs[1]
 dx_img=np.zeros((imgx-1,imgy-1))
 dy_img=np.zeros((imgx-1,imgy-1))
 dxy_img=np.zeros((imgx-1,imgy-1))
 dyx_img=np.zeros((imgx-1,imgy-1))
 for y in range(1,imgy-1):
  for x in range(1,imgx-1):
   dx_img[x-1,y-1]=np_imgx[x+1,y]-np_imgx[x-1,y]
   dy_img[x-1,y-1]=np_imgx[x,y+1]-np_imgx[x,y-1]
   dxy_img[x-1,y-1]=np_imgx[x+1,y+1]-np_imgx[x-1,y-1]
   dyx_img[x-1,y-1]=np_imgx[x-1,y+1]-np_imgx[x+1,y+1]
 total_diff = np.abs(dx_img) + np.abs(dy_img) + np.abs(dxy_img) + np.abs(dyx_img)
 total_diff = ( total_diff*255 )/( total_diff.max() - total_diff.min() )
 return {'t':total_diff.copy(),'x':dx_img.copy(),'y':dy_img.copy(),'xy':dxy_img.copy(),'yx':dyx_img.copy()}
def imgShow(np_img,cmap='gray',figsize=(10,5),dpi=80):
 plt.figure(figsize=figsize,dpi=dpi)
 plt.imshow(np_img,cmap=cmap)
 plt.show()
def rotateImage(img,angle,expand=1):
 yh=img.sum(axis=0)
 xh=img.sum(axis=1)
 yc=np.argmax(yh)
 xc=np.argmax(xh)
 #center=(0,0)
 center=(xc,yc)
 src_im = Image.fromarray(np.uint8(img))
 rs = src_im.size
 diag = np.round(np.power(rs[0]**2+rs[1]**2,0.5),0)
 rotW=int(diag)
 rotH=rotW

 dst_im = Image.new('L', (rotW,rotH), (0) )
 rot = src_im.rotate( angle, expand=expand)
 rs = rot.size
 posX = int((rotW-rs[0])/2)
 posY = int((rotH-rs[1])/2)

 dst_im.paste( rot, (posX, posY) ,mask=rot.split()[0])
 rot_im=np.asarray(dst_im)
 dst_im.close()
 src_im.close()
 return rot_im.copy()
def expFilter(inp,amplitude=255,factor=0.5,shift=0.5,range_s=0,range_e=255):
 """
 To minimize noise with low magnitude in a picture, an exponential filter could transform intensities  
 according to a threshold (TH). For this function, "Shift" (S) moves the transformation TH to the right as  
 it increases. "Factor" (F) moves from a linear transformation to a step as it increases.
 amplitude: is the factor to multiply the filter. For images, the value is 255.
 range_s and  range_e are the minimun and maximun values that contains the input matrix. For images these
 values are 0 and 255 respectively.
 """
 factor=factor
 shift = shift
 x=inp.copy()
 b=int((range_e - range_s)*shift)
 y=amplitude/(1+np.exp(-(x-b+0.01)*factor))
 return y.copy()
def resizeImg(np_img,w=300,h=300):
 img = Image.fromarray(np.uint8(np_img))
 img = img.resize((w, h),Image.ANTIALIAS)
 return np.asarray(img).copy()
def printPlot(message,img,v=False,plotAll=False):
 if v==True:
  print(message)
 if plotAll==True:
  image_proc.imgShow(img)
def imageGrayBorderRotate(imgName,borderT='t',rotateRange=(1,360,45),rw=300,rh=300,v=False,plotAll=False,expFilterP={'factor':0.5,'shift':0.1}):
 img_gray = imgToGray(imgName)
 message=imgName+' '+"Now gray"
 printPlot(message,img_gray,v=v,plotAll=plotAll)
 img_border=borderFilter(img_gray)
 #['y', 'xy', 'x', 'yx', 't']
 img_border=img_border[borderT]
 message=imgName+' '+"borders detected"
 printPlot(message,img_border,v=v,plotAll=plotAll)
 img_exp = expFilter(img_border,factor=expFilterP['factor'],shift=expFilterP['shift'])
 message=imgName+' '+"passed through exp filter"
 printPlot(message,img_exp,v=v,plotAll=plotAll)
 
 angle_range = np.arange(rotateRange[0],rotateRange[1],rotateRange[2])
 img_arr = np.zeros((angle_range[:].shape[0],rw*rh))
 
 for i,angle in enumerate(angle_range):
  img_rot=rotateImage(img_exp,angle=angle,expand=1)
  message=imgName+' '+"Rotated: "+str(angle)+" degrees"
  printPlot(message,img_rot,v=v,plotAll=plotAll)
  img_res=resizeImg(img_rot,w=rw,h=rh)
  img_arr[i]=img_res.flatten()
 return img_arr
def dfFileTargetToArray(df,rh=300,rw=300,v=False,plotAll=False,borderT='t',rotateRange=(1,90,10),expFilterP={'factor':0.5,'shift':0.1}):
 """
 The function dfFileTargetToArray gets the filepath form a pandas dataframe in the field file
 and creates a group of numpy samples with the border detection using the t type. Creates samples
 rotated according to the rotateRange param. expFilterP have the paremeters to the exponential filter.
 v if True, shows messages according to the progress. plotAll if True plots every image it
 goes creating. rh and rw are the heigth and width to resize the images.
 The ouput is an flatten numpy array with a +1 column, the target class from the target column specified
 in the dataframe.
 """
 dfr=df.shape[0]
 np_xy=np.zeros((0,rh*rw+1))
 for i in range(0,dfr,1):
  if v==True:
   print("Processing file:",str(i),"from",dfr)
  dfx=df.iloc[i]
  img_test=dfx['file']  
  img_arr_x=imageGrayBorderRotate(imgName=img_test,borderT=borderT,rotateRange=rotateRange,v=v,plotAll=plotAll,rw=rw,rh=rh,expFilterP=expFilterP)
  npxsr=img_arr_x.shape[0]
  npx=np.zeros((npxsr,1))
  npx[:]=dfx['target']
  np_xy_x = np.append(img_arr_x,npx,axis=1)
  np_xy=np.append(np_xy,np_xy_x,axis=0)
 return np_xy.copy()
