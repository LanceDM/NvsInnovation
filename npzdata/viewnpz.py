import numpy as np

# Load the NPZ file
npz_file = np.load("image_skeleton_data.npz", allow_pickle=True)

# List all keys inside the NPZ file
print("Keys in NPZ file:", npz_file.files)

# Load skeleton data
skeleton_data = npz_file["data"]

# Check if the data is stored as an object array
if skeleton_data.dtype == object:
    print(f"Skeleton Data Length (Frames): {len(skeleton_data)}")
else:
    print(f"Skeleton Data Shape: {skeleton_data.shape}")  # Expected: (num_frames, max_people, 33, 2)

# Print all frames and all skeletons
for frame_idx, frame in enumerate(skeleton_data):
    print(f"\nFrame {frame_idx + 1}:")
    
    if isinstance(frame, list):  # If frame contains multiple people
        for person_idx, person in enumerate(frame):
            print(f"  Person {person_idx + 1} Skeleton Data:\n", np.array(person))  # Convert to array for readability
    else:
        print("  Single Person Skeleton Data:\n", np.array(frame))  # Print single person skeleton


# import numpy as np

# # Load the NPZ file
# npz_file = np.load("multi_person_skeleton_image.npz", allow_pickle=True)

# # Print raw NPZ file contents
# print("Raw NPZ File Data:")
# for key in npz_file.files:
#     print(f"\nKey: {key}")
#     print(npz_file[key])  # Print entire content of each stored key
