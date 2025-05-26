import matplotlib.pyplot as plt

# Provided paths_dict data
from paths import paths_dict

plt.figure(figsize=(10, 8))
for radial, coords in paths_dict.items():
    lats, lons = zip(*coords)
    plt.plot(lons, lats, label=f'Radial {radial}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Paths by Radial')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()