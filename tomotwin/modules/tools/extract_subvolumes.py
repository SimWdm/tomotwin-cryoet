import argparse
from argparse import ArgumentParser

import mrcfile
import numpy as np
import os
import pandas as pd
import shutil
import tqdm
import zarr

from typing import List

from tomotwin.modules.common.dog import get_dog_proposals
from tomotwin.modules.common.extract_subvols import extract_subvols
from tomotwin.modules.tools.tomotwintool import TomoTwinTool

#%%
class ExtractSubvolumes(TomoTwinTool):
    '''
    Extracts a subvolume (reference) from a volume and save it to disk.
    '''

    def get_command_name(self) -> str:
        '''
        :return: Command name
        '''
        return "extract_subvolumes"

    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Extracts subvolumes (e.g. for test-time-training).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument('--vol_files', nargs='+', type=str, required=True, 
                            help='List of volumes to extract from')
        parser.add_argument('--dog_sigmas', type=str, required=False, default=None,
                            help='List of sigma values for the DoG filter to use for particle detection. If not provided, no DoG filtering is applied.')
        parser.add_argument('--coord_files', type=List, required=False, default=None,
                            help='List of coordinates files with coordinates of particles to extract (X, Y, Z). Each file corresponds to a volume in --vol_files.')
        parser.add_argument('--size', type=int, default=37,
                            help='Box size. For TomoTwin this should be typically a size of 37.')
        parser.add_argument('--out', type=str, required=True,
                            help='output path')
        parser.add_argument('-y', '--yes', action='store_true', default=False,
                            help='Automatically answer yes to any prompts, e.g. overwriting existing files.')


        return parser

    @staticmethod
    def extract_and_save(volume: np.array, positions: pd.DataFrame, box_size: int, out_pth: str, basename: str, apix=None) -> List[str]:
        '''
        :param volume: Volume from which the the references should be extracted
        :param positions: Dataframe with coordinates
        :param box_size: Extraction boxsize in pixel
        :param out_pth: Folder where the subvolumes are written
        :param basename: Basename for files, index and filename are added automatically.
        :return: List with paths of written subvolumes.
        '''

        files_written = []
        for index, row in tqdm.tqdm(positions.iterrows()):
            x = row['X']
            y = row['Y']
            z = row['Z']
            odd_factor = box_size % 2 
            # Define corners of box
            nx1 = (x - (box_size - odd_factor) // 2)
            nx2 = (x + (box_size - odd_factor) // 2 + odd_factor)
            ny1 = (y - (box_size - odd_factor) // 2)
            ny2 = (y + (box_size - odd_factor) // 2 + odd_factor)
            nz1 = (z - (box_size - odd_factor) // 2)
            nz2 = (z + (box_size - odd_factor) // 2 + odd_factor)


            subvol = volume[int(nz1): int(nz2), int(ny1): int(ny2), int(nx1): int(nx2)]

            if subvol.shape != (box_size, box_size, box_size):
                continue
            subvol = -1 * subvol  # invert
            subvol = subvol.astype(np.float32)
            fname = os.path.join(out_pth,f"{basename}_{index}.mrc")
            with mrcfile.new(fname) as newmrc:
                newmrc.set_data(subvol)
                if apix:
                    newmrc.voxel_size = apix
            files_written.append(fname)
        return files_written
    

    def run(self, args):
        vol_files = args.vol_files
        dog_sigmas = args.dog_sigmas
        coord_files = args.coord_files
        boxsize = args.size
        path_output = args.out
        
        
        # make sure dog_sigmas and coord_files are mutually exclusive, but not both None
        if dog_sigmas is not None and coord_files is not None:
            raise ValueError("Cannot use both --dog_sigmas and --coord_files. Please provide only one of them.")
        if coord_files is None:
            if dog_sigmas is None:
                raise ValueError("Please provide either --dog_sigmas or --coord_files. Both cannot be None.")
        
        if dog_sigmas is not None:
            dog_sigmas = [int(x) for x in dog_sigmas.split(',')]
        

        
        # if path_out exists, ask user if they want to overwrite
        if os.path.exists(path_output):
            overwrite = input(f"Output path {path_output} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("Exiting without overwriting.")
                return
            else:
                print(f"Overwriting {path_output}...")
                shutil.rmtree(path_output)
        os.makedirs(path_output)    

        
        subtomo_idx = 0
        for vol_idx, vol_file in enumerate(vol_files):
            
            
            if vol_file.endswith('.zarr'):
                vol = zarr.load(vol_file)[0]
            elif vol_file.endswith('.mrc'):
                with mrcfile.open(vol_file) as mrc:
                    vol = mrc.data
            else:
                raise ValueError(f"Unsupported file format: {vol_file}.")
                                        
            if coord_files is not None:
                coords = pd.read_csv(coord_files[vol_idx])
                coords = coords[["X", "Y", "Z"]]
                
            else:
                if dog_sigmas is not None:
                    coords = get_dog_proposals(vol, sigmas=dog_sigmas, downsampling=2)
            
            subvols = extract_subvols(
                vol=vol, 
                subvol_center_coords=coords, 
                subvol_size=boxsize, 
                skip_out_of_bounds=True 
            )
            
            for subvol in tqdm.tqdm(subvols, "Extracting subvolumes from volume"):
                subvol_file = f"{path_output}/{os.path.splitext(os.path.basename(vol_file))[0]}_0XXX_{subtomo_idx}.mrc"
                with mrcfile.new(subvol_file, overwrite=True) as new_mrc:
                    new_mrc.set_data(-1 * subvol)
                subtomo_idx += 1

        print(f"Wrote {len(os.listdir(path_output))} subvolumes to {path_output}.")


# %%
