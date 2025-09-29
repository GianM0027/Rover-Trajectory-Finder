# path_solver_minimal.py

from hirise_dtm import HiriseDTM
import numpy as np
from collections import deque

DTM_PATH = "./DTMs/DTEED_082989_1630_083055_1630_A01.IMG"

def generate_start_end_map(dim=10):
    position_map = np.zeros((dim, dim), dtype=float)
    start_pos_flat, end_pos_flat = np.random.choice(dim * dim, size=2, replace=False)
    
    start_row, start_col = divmod(start_pos_flat, dim)
    end_row, end_col = divmod(end_pos_flat, dim)
    
    position_map[start_row, start_col] = 1.0
    position_map[end_row, end_col] = -1.0
    
    return position_map

def find_path(position_map, altitude_map, min_altitude_diff = -1.0, max_altitude_diff=1.0):
    try:
        start_pos = tuple(np.argwhere(position_map == 1.0)[0])
        end_pos = tuple(np.argwhere(position_map == -1.0)[0])
    except IndexError:
        return None

    dim = position_map.shape[0]
    queue = deque([[start_pos]])
    visited = {start_pos}
    
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        
        if (r, c) == end_pos:
            return path
            
        current_altitude = altitude_map[r, c]
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < dim and 0 <= nc < dim and (nr, nc) not in visited:
                neighbor_altitude = altitude_map[nr, nc]

                #print(f"Current: ({r},{c}) Alt: {current_altitude:.2f} -> Neighbor: ({nr},{nc}) Alt: {neighbor_altitude:.2f} | Diff: {neighbor_altitude - current_altitude:.2f}")
                
                if min_altitude_diff < abs(current_altitude - neighbor_altitude) < max_altitude_diff:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(new_path)
                    
    return None

def display_map(position_map, path=None):
    vis_map = np.full(position_map.shape, '.', dtype='<U3')
    vis_map[position_map == 1.0] = 'S'
    vis_map[position_map == -1.0] = 'E'

    if path:
        for r, c in path:
            if vis_map[r, c] not in ['S', 'E']:
                vis_map[r, c] = '*'
    
    print("\n--- Map Visualization ---")
    for row in vis_map:
        print(" ".join(f"{x:>3}" for x in row))

def main():
    maps_dim = 10
    try:
        hirise = HiriseDTM(img_path=DTM_PATH)
        altitude_portion, _ = hirise.get_portion_of_map(size=maps_dim)
        position_map = generate_start_end_map(dim=maps_dim)
        
        found_path = find_path(position_map, altitude_portion,min_altitude_diff=-0.8, max_altitude_diff=1.0)


        #print("Altitude Map:")
        #print(altitude_portion)

        print("\n--- Position Matrix (S=Start, E=End) ---")
        display_map(position_map)
        
        if found_path:
            print("Path found! ✅")
            display_map(position_map, found_path)
        else:
            print("No valid path found. ❌")
            display_map(position_map) # Mostra la mappa anche se non trova il percorso

    except FileNotFoundError:
        print(f"Error: The file was not found at '{DTM_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()