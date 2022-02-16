import numpy as np

arr = np.array([
    [
        [1, 1],
        [2, 2]
    ],
    [
        [2, 2],
        [3, 3]
    ],
    [
        [3, 3],
        [4, 4]
    ],
    [
        [4, 4],
        [5, 5]
    ]
])


item = np.array([
    [5, 5],
    [6, 6]
])

print(arr.shape)
print(item.shape)

append = np.append(item.reshape(1, 2, 2), arr[:3,:,:], axis=0)
print(append)
append = np.append(item.reshape(1, 2, 2), append[:3,:,:], axis=0)
print(append)
append = np.append(item.reshape(1, 2, 2), append[:3,:,:], axis=0)
print(append)
append = np.append(item.reshape(1, 2, 2), append[:3,:,:], axis=0)

print(append)