#!/usr/bin/env python
# coding: utf-8

# In[1]:


with open("./src.txt", "r") as f:
    src = list(map(int, f.readlines()))
with open("./dst.txt", "r") as f:
    dst = list(map(int, f.readlines()))


# In[2]:


import numpy as np
img_in = np.array(src, dtype=np.uint8).reshape(128, 128)
img_out = np.array(dst).reshape(126, 126)
img_in, img_out


# In[3]:


import plotly.graph_objs as go
import plotly.offline as py
trace = go.Heatmap(
        z=np.flip(img_in, 0),
        y=np.array(range(128)),
        x=np.array(range(128))
    )
py.iplot(dict(data=trace))


# In[4]:


from matplotlib import pyplot as plt
plt.imshow(img_in, cmap='gray')


# In[5]:


plt.imshow(img_out, cmap='gray')


# In[6]:


import cv2
import time

start_time = time.time()
sobel_x = cv2.Sobel(img_in,cv2.CV_8U,1,0)
end_time = time.time()

print("SW Time: {}s".format(end_time-start_time))

start_time = time.time()
sobel_y = cv2.Sobel(img_in,cv2.CV_8U,0,1)
end_time = time.time()

print("SW Time: {}s".format(end_time-start_time))

fig_sobel3 = plt.figure()
fig_sobel3.set_figheight(4)
fig_sobel3.set_figwidth(15)
fig_sobel3.add_subplot(131)
plt.imshow(sobel_x, cmap='gray')
fig_sobel3.add_subplot(132)
plt.imshow(sobel_y, cmap='gray')


# In[27]:


from pynq import Overlay, allocate
overlay = Overlay("./overlay/filter2d.bit")
filter2d = overlay.filter2d_accel_0


# In[22]:


overlay.hierarchy_dict


# In[14]:


img_in_buffer = allocate(shape=(len(src),), dtype='i4')
img_out_buffer = allocate(shape=(len(dst),), dtype='i4')
kernel1_buffer = allocate(shape=(3*3,), dtype='i4')


# In[16]:


np.copyto(img_in_buffer, np.int16(src))

np.copyto(kernel1_buffer, np.int16([-1, -2, -1, 
                                   0,  0,  0, 
                                   1,  2,  1]))


# In[28]:


filter2d.s_axi_control.write(0x10, img_in_buffer.physical_address)
filter2d.s_axi_control.write(0x28, img_out_buffer.physical_address)
filter2d.s_axi_control.write(0x10, 128)
filter2d.s_axi_control.write(0x18, 128)


# In[34]:


import time

filter2d.s_axi_control.write(0x1c, kernel1_buffer.physical_address)
filter2d.s_axi_control.write(0x00, 0x01)
start_time = time.time()
while True:
    reg = filter2d.s_axi_CTRL.read(0x00)
    if reg != 1:
        break
end_time = time.time()

print("HW Time：{}s".format(end_time - start_time))

img_out_x = np.int16(img_out_buffer)



# In[ ]:




