from draw import *
import cairo
import numpy as np
from skimage.color import rgba2rgb,rgb2gray
from math import pi

def rgba2gray(x):
    return rgb2gray(rgba2rgb(x))

def makeSurface(size):
	img = cairo.ImageSurface(cairo.FORMAT_ARGB32,size,size)	
	return img

def makeContext(img):
	return cairo.Context(img)

def makeCanvas(imsize,colour):
    img = makeSurface(imsize)
    ctx = makeContext(img)
    # ctx.set_source_rgba(1.,1.,1.,1.0)
    # if colour:
    rgba = np.random.rand(4,1)
    ctx.set_source_rgba(rgba[0],rgba[1],rgba[2],rgba[3])
    ctx.paint()
    return img,ctx,

def clearContext(ctx,colour):
    # ctx.set_source_rgba(0.,0.,0.,1.0)
    # if colour:
    rgb = np.random.rand(3,1)
    ctx.set_source_rgba(rgb[0],rgb[1],rgb[2],1.0)
    ctx.paint()
    return ctx

def randomTransformations(ctx, size, colour):
    ctx = translateShape(ctx,0,0)
    ctx = translateShape(ctx,np.random.randint(0,
            round(0.5*size)),np.random.randint(0,
            round(0.5*size)))
    ctx = rotateShape(ctx,pi*np.random.rand())	
    ctx = scaleShape(ctx,[1.9*np.random.rand()+0.1,1.9*np.random.rand()+0.1])
    # ctx = scaleShape(ctx,[0.01*size*np.random.rand(), 0.2*size*np.random.rand()])
    # rgba = [0.5,0.5,0.5,np.random.rand()]
    # if colour:
    rgba = np.random.rand(4,1)
    ctx = colouriseShape(ctx,rgba)
    return ctx

def draw(ctx, size, minmax, colour):
    print('-----new image----')
    num_objects = np.random.randint(minmax[0],minmax[1])
    print("Total objects: ", num_objects)
    for i in range(num_objects):
        ctx = randomTransformations(ctx,size,colour)
        temp = np.random.randint(0,4)
        centre = [0.,0.]
        shape_size = 50 # np.random.randint(round(0.05*size),
            # round(0.5*size))
        if temp==0:				
            print("Added Rect")
            ctx = drawRect(ctx,centre,shape_size,[0.25+3.75*np.random.rand(), 0.25+3.75*np.random.rand()])		
        elif temp==1:
            print("Added Polygon")
            ctx = drawPolygon(ctx,centre,shape_size,numVertices=round(np.random.randint(3,9)))
        elif temp==2:
            print("Added Star")
            ctx = drawStar(ctx,centre,shape_size,np.random.randint(round(0.1*shape_size),round(0.8*shape_size)),numVertices=round(np.random.randint(3,9)))	
        elif temp==3:
            print("Added Circle")
            ctx = drawCircle(ctx,centre,shape_size)
        elif temp==4:
            print("Added Ellipse")
            ctx = drawEllipse(ctx,centre,shape_size,[0.25+3.75*np.random.rand(), 0.25+3.75*np.random.rand()])
        
        ctx.set_operator(cairo.Operator.OVER)
        ctx.fill()
    return ctx

def makeImage(img,ctx,size,minmax,colour):
    ctx = draw(ctx,size,minmax,colour)
    tmp = np.frombuffer(img.get_data(),np.uint8)
    tmp.shape = [size,size,4]
    return ctx,tmp,img 

def makeDataset(size: int, num_images: int, min_objects: int, max_objects: int, colour: bool):
    """
    draws stimuli for all requested combinations of parameters
    and returns a dictionary with numpy arrays
    """
    img,ctx = makeCanvas(size, colour)
    img_centre = np.floor(img.get_width()/2), np.floor(img.get_height()/2)
    # make array with all combinations of feature levels 
    # make loop-up dictionary with feature values within requested range for each feature dimension
    # if colour:
    all_IMGs = np.zeros((num_images,size,size,4),np.uint8)
    # else:
        # all_IMGs = np.zeros((num_images,size,size),np.uint8)
    # iterate through shapes and exemplars, generate one image per iteration (TODO: parallelise with second for loop)
    for i in range(num_images):
        ctx.save()
        ctx, img_mat, img = makeImage(img, ctx, size, [min_objects, max_objects],colour)
        all_IMGs[i,...] = img_mat
        ctx.restore()
        ctx = clearContext(ctx,colour)

    if not colour:
        a = np.zeros((all_IMGs.shape[0],all_IMGs.shape[1],all_IMGs.shape[1]))
        for i in range(all_IMGs.shape[0]):
            a[i,...] = rgba2gray(all_IMGs[i,...])
        
        return a
    return all_IMGs