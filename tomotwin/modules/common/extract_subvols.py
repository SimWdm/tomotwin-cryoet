import pandas as pd

def extract_subvols(vol, subvol_center_coords, subvol_size=37, skip_out_of_bounds=False):
    """
    Extrac
    
    """
    if isinstance(subvol_center_coords, pd.DataFrame):
        if "x" in subvol_center_coords.columns and "y" in subvol_center_coords.columns and "z" in subvol_center_coords.columns:
            subvol_center_coords = subvol_center_coords[["z", "y", "x"]].values
        elif "X" in subvol_center_coords.columns and "Y" in subvol_center_coords.columns and "Z" in subvol_center_coords.columns:
            subvol_center_coords = subvol_center_coords[["Z", "Y", "X"]].values
    
    # check if subvol_size is even 
    if subvol_size % 2 == 1:
        lower = subvol_size//2
        upper = subvol_size//2 + 1
    else:
        lower = upper = subvol_size//2        
    
    subvols = []
    skip = 0
    for z, y, x in subvol_center_coords:
        z, y, x = int(z), int(y), int(x)
        if any([x-lower < 0, y-lower < 0, z-lower < 0, x+upper > vol.shape[2], y+upper > vol.shape[1], z+upper > vol.shape[0]]):
            if skip_out_of_bounds:
                skip += 1
                continue
            else:
                raise ValueError(f"subvol at coordinates (z={z}, y={y}, x={x}) is out of bounds.")
        subvol = vol[z-lower:z+upper, y-lower:y+upper, x-lower:x+upper]
        subvols.append(subvol)

    return subvols