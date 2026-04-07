import os
import trimesh
import threading
from queue import Queue
import csv
import statistics

"""Scan the user's Desktop for .ply files, load each with trimesh one at a time,
analyze connected components, compute statistics, and write to CSV.
"""


def analyze_ply_components(ply_path, timeout=30):
    """Process a single file with timeout protection and return component statistics"""
    result_queue = Queue()
    
    def worker():
        try:
            mesh = trimesh.load(ply_path, force='mesh')
        except Exception as e:
            import traceback
            error_msg = f"Failed to load '{ply_path}': {type(e).__name__}: {e}"
            result_queue.put(('error_load', error_msg, traceback.format_exc()))
            return

        if mesh.is_empty:
            result_queue.put(('success', ply_path, []))
            return

        # trimesh objects have split method that returns connected pieces
        try:
            components = mesh.split(only_watertight=False)
            stats = []
            for comp in components:
                vol = comp.volume if hasattr(comp, 'volume') else 0
                area = comp.area if hasattr(comp, 'area') else 0
                faces = len(comp.faces) if hasattr(comp, 'faces') else 0
                stats.append((vol, area, faces))
            result_queue.put(('success', ply_path, stats))
        except Exception as e:
            # fallback: use graph connectivity on vertices
            try:
                graph = mesh.vertex_adjacency_graph
                # connected_components returns list of sets
                import networkx as nx
                comp_sets = list(nx.connected_components(graph))
                # For fallback, we can't easily get volume/area/faces, so just count
                stats = [(0, 0, 0)] * len(comp_sets)  # placeholder
                result_queue.put(('success', ply_path, stats))
            except Exception as fallback_e:
                import traceback
                error_msg = f"Could not determine components for '{ply_path}': {type(e).__name__}/{type(fallback_e).__name__}"
                result_queue.put(('error_comp', error_msg, traceback.format_exc()))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return ('timeout', f"File processing timed out after {timeout}s: '{ply_path}'", [])
    
    if result_queue.empty():
        return ('error', f"No result for '{ply_path}'", [])
    
    return result_queue.get()


def compute_stats(values):
    """Compute min, median, max for a list of values"""
    if not values:
        return None, None, None
    sorted_vals = sorted(values)
    return min(sorted_vals), statistics.median(sorted_vals), max(sorted_vals)


def main():
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    print(f"Scanning desktop directory: {desktop}\n")

    ply_files = []
    for root, dirs, files in os.walk(desktop):
        for f in files:
            if f.lower().endswith('.ply'):
                ply_files.append(os.path.join(root, f))

    if not ply_files:
        print("No .ply files found on the desktop.")
        return

    csv_path = os.path.join(os.path.dirname(__file__), 'ply_component_stats.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'file', 'total_components',
            'avg_volume', 'avg_area', 'avg_faces',
            'min_volume', 'median_volume', 'max_volume',
            'min_area', 'median_area', 'max_area',
            'min_faces', 'median_faces', 'max_faces',
            'avg_volume_excl_largest', 'avg_area_excl_largest', 'avg_faces_excl_largest'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ply in sorted(ply_files):
            result_type, message, stats = analyze_ply_components(ply, timeout=30)
            
            if result_type == 'success':
                ply_path, comp_stats = message, stats
                if not comp_stats:
                    # Empty mesh
                    row = {
                        'file': ply,
                        'total_components': 0,
                        'avg_volume': 0, 'avg_area': 0, 'avg_faces': 0,
                        'min_volume': 0, 'median_volume': 0, 'max_volume': 0,
                        'min_area': 0, 'median_area': 0, 'max_area': 0,
                        'min_faces': 0, 'median_faces': 0, 'max_faces': 0,
                        'avg_volume_excl_largest': 0, 'avg_area_excl_largest': 0, 'avg_faces_excl_largest': 0
                    }
                else:
                    volumes = [s[0] for s in comp_stats]
                    areas = [s[1] for s in comp_stats]
                    faces = [s[2] for s in comp_stats]
                    
                    total_comp = len(comp_stats)
                    avg_vol = sum(volumes) / total_comp
                    avg_area = sum(areas) / total_comp
                    avg_faces = sum(faces) / total_comp
                    
                    min_vol, med_vol, max_vol = compute_stats(volumes)
                    min_area, med_area, max_area = compute_stats(areas)
                    min_faces, med_faces, max_faces = compute_stats(faces)
                    
                    # Exclude largest by volume
                    if total_comp > 1:
                        max_vol_idx = volumes.index(max_vol)
                        excl_volumes = volumes[:max_vol_idx] + volumes[max_vol_idx+1:]
                        excl_areas = areas[:max_vol_idx] + areas[max_vol_idx+1:]
                        excl_faces = faces[:max_vol_idx] + faces[max_vol_idx+1:]
                        avg_vol_excl = sum(excl_volumes) / len(excl_volumes)
                        avg_area_excl = sum(excl_areas) / len(excl_areas)
                        avg_faces_excl = sum(excl_faces) / len(excl_faces)
                    else:
                        avg_vol_excl = avg_vol
                        avg_area_excl = avg_area
                        avg_faces_excl = avg_faces
                    
                    row = {
                        'file': ply,
                        'total_components': total_comp,
                        'avg_volume': avg_vol, 'avg_area': avg_area, 'avg_faces': avg_faces,
                        'min_volume': min_vol, 'median_volume': med_vol, 'max_volume': max_vol,
                        'min_area': min_area, 'median_area': med_area, 'max_area': max_area,
                        'min_faces': min_faces, 'median_faces': med_faces, 'max_faces': max_faces,
                        'avg_volume_excl_largest': avg_vol_excl, 'avg_area_excl_largest': avg_area_excl, 'avg_faces_excl_largest': avg_faces_excl
                    }
                writer.writerow(row)
                print(f"Processed: {ply} - {row['total_components']} components")
            elif result_type == 'timeout':
                print(f"⏱️  TIMEOUT (30s): {ply}")
                # Write a row with N/A or something
                row = {'file': ply, 'total_components': 'TIMEOUT'}
                for fn in fieldnames[1:]:
                    row[fn] = 'N/A'
                writer.writerow(row)
            else:
                print(f"❌ {message}")
                row = {'file': ply, 'total_components': 'ERROR'}
                for fn in fieldnames[1:]:
                    row[fn] = 'N/A'
                writer.writerow(row)

    print(f"\nStats written to: {csv_path}")


if __name__ == '__main__':
    main()
