def patching(dims, step, tile_dim):
    y_dim,x_dim=dims
    for y in range(0, y_dim, step):
        for x in range(0, x_dim, step):
            x_new = x_dim-tile_dim if x+tile_dim>x_dim else x
            y_new = y_dim-tile_dim if y+tile_dim>y_dim else y
            yield x_new, y_new


    def predict(wsi):
    #y_dim, x_dim, _ = image.shape
    LEVEL=2
    tile_dim=1600
    ds_factor=10
    wsi_dims = slide.level_dimensions[LEVEL]
    step=600
    margin=int((tile_dim-step)/2)
    h, w = wsi_dims
    h, w = h/10, w/10
    c=Canvas(w,h)

    for x, y in patching(wsi_dims, step, tile_dim):
        tile = slide.read_region((y*2**LEVEL,x*2**LEVEL),LEVEL,(tile_dim,tile_dim))
        tile=cv2.resize(np.array(tile.convert('RGB')), (160,160),interpolation = cv2.INTER_AREA)
        tile=tile.astype('uint8')
        print(w-tile_dim/10)
        stitch(
            c,
            tile, 
            int(x//ds_factor), 
            int(y//ds_factor), 
            h,
            w,
            int(tile_dim/ds_factor),
            int(step/ds_factor), 
            int(margin/ds_factor)
        )     
        plt.imshow(c.canvas)
        plt.show()