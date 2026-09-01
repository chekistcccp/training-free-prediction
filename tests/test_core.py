import numpy as np
from PIL import Image
from src.data import resize_mask,tumor_slice_labels,window_ct
from src.perturb import perturb_image
from src.utils import Box,grid_box,parse_grid_cells

def test_window_ct_range():
    y=window_ct(np.array([-1000,-160,40,240,1000],dtype=np.float32),40,400); assert y.dtype==np.uint8 and int(y.min())==0 and int(y.max())==255

def test_slice_labels():
    m=np.zeros((5,4,4),bool); m[2,1:3,1:3]=True; assert tumor_slice_labels(m).tolist()==[0,0,1,0,0]

def test_grid_parser():
    assert grid_box(400,400,4,1,2)==Box(200,100,300,200); assert parse_grid_cells('{"candidate_cells":["B3","C2"]}',4)==['B3','C2']

def test_perturbations():
    a=np.full((128,128,3),100,np.uint8); a[32:64,32:64]=220; img=Image.fromarray(a); b=Box(32,32,64,64)
    for k in ['blur','mean_fill','inpaint']: assert perturb_image(img,b,k).size==img.size

def test_resize_mask():
    m=np.zeros((10,10),bool); m[2:5,3:7]=True; r=resize_mask(m,32); assert r.shape==(32,32) and r.any()
