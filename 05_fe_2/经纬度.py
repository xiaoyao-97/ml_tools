import reverse_geocoder as rg

def get_location(lat, long):
    coordinates = (lat, long)
    result = rg.search(coordinates)
    if result:
        return result[0]['name'], result[0]['admin1'], result[0]['cc']
    else:
        return "Unknown location"

latitude = 37.7749  # 纬度
longitude = -122.4194  # 经度

location, state, country = get_location(latitude, longitude)
print(f"Location: {location}, State: {state}, Country: {country}")


"""更简单的：
coordinates = list(zip(df['Latitude'], df['Longitude']))
results = rg.search(coordinates)
"""

"""旋转：
df['rot_15_x'] = (np.cos(np.radians(15)) * df['Longitude']) + \
                  (np.sin(np.radians(15)) * df['Latitude'])
    
df['rot_15_y'] = (np.cos(np.radians(15)) * df['Latitude']) + \
                  (np.sin(np.radians(15)) * df['Longitude'])
    
df['rot_30_x'] = (np.cos(np.radians(30)) * df['Longitude']) + \
                  (np.sin(np.radians(30)) * df['Latitude'])
    
df['rot_30_y'] = (np.cos(np.radians(30)) * df['Latitude']) + \
                  (np.sin(np.radians(30)) * df['Longitude'])
    
df['rot_45_x'] = (np.cos(np.radians(44)) * df['Longitude']) + \
                  (np.sin(np.radians(45)) * df['Latitude'])
"""


"""zip code to lat lon
import pgeocode

nomi = pgeocode.Nominatim('fr')  # 设置为法国
query = nomi.query_postal_code(['75008','75002'])  # 示例：巴黎的一个邮政编码

d = {
    "lat": query["latitude"],
    "lon": query["longitude"]
}

print(d)"""