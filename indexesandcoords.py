def indextocoords(shape, index):
    _, Y, Z = shape
    
    x = index // (Y*Z)
    y = (index % (Y*Z))//Z
    z = index % Z
    
    return (x, y, z)

def coordstoindex(shape, coords):
    _, Y, Z = shape
    x, y, z = coords
    
    return (x * (Y*Z)) + (y*Z + z)