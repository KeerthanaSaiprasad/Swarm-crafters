import os
import requests
import json
from typing import Dict

def find_similar_nearby_places(input_place_name: str) -> Dict:
    """
    Finds similar nearby places based on an input place name.
    
    Args:
        input_place_name (str): The name of the place to search around (e.g., "Manis Dum Biryani, Siruseri")
    
    Returns:
        dict: A dictionary containing:
            - 'original_place': Details of the input place
            - 'nearby_places': List of similar nearby places
            - 'success': Boolean indicating if the operation was successful
            - 'error': Error message if operation failed
    """
    
    # Configuration
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg")
    search_radius = 5000
    
    def get_coordinates_from_address(address):
        """Get latitude and longitude for a given address using Google Geocoding API."""
        base_url = "https://maps.googleapis.com/maps/api/geocode/json?"
        params = {"address": address, "key": api_key}

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK" and data["results"]:
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
            elif data["status"] == "ZERO_RESULTS":
                return None
            else:
                return None

        except requests.exceptions.RequestException as e:
            return None
        except json.JSONDecodeError:
            return None

    def get_place_details_by_name(place_name, location_bias=None):
        """Find a place by its name and return its details."""
        base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?"
        
        find_place_params = {
            "input": place_name,
            "inputtype": "textquery",
            "fields": "place_id,name,geometry,types",
            "key": api_key,
        }
        
        if location_bias:
            find_place_params["locationbias"] = location_bias

        try:
            response = requests.get(base_url, params=find_place_params)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK" and data["candidates"]:
                candidate = data["candidates"][0]
                place_id = candidate["place_id"]

                details_url = "https://maps.googleapis.com/maps/api/place/details/json?"
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,geometry,types,price_level,rating,user_ratings_total,url,website",
                    "key": api_key,
                }
                details_response = requests.get(details_url, params=details_params)
                details_response.raise_for_status()
                details_data = details_response.json()

                if details_data["status"] == "OK":
                    result = details_data["result"]
                    return {
                        "name": result.get("name"),
                        "place_id": result.get("place_id"),
                        "address": result.get("formatted_address"),
                        "latitude": result["geometry"]["location"]["lat"],
                        "longitude": result["geometry"]["location"]["lng"],
                        "types": result.get("types", []),
                        "price_level": result.get("price_level"),
                        "rating": result.get("rating"),
                        "user_ratings_total": result.get("user_ratings_total"),
                        "Maps_url": result.get("url"),
                        "website": result.get("website")
                    }
                else:
                    return None
            elif data["status"] == "ZERO_RESULTS":
                return None
            else:
                return None

        except requests.exceptions.RequestException as e:
            return None
        except json.JSONDecodeError:
            return None

    def find_nearby_places_by_type_and_cost(latitude, longitude, search_types, radius, max_price_level=None, min_price_level=None, exclude_place_id=None):
        """Find nearby places based on specified types and optional price levels."""
        base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "key": api_key,
        }

        if search_types:
            params["type"] = search_types[0]
            if len(search_types) > 1:
                params["keyword"] = " ".join(search_types[1:])

        if min_price_level is not None:
            params["minprice"] = min_price_level
        if max_price_level is not None:
            params["maxprice"] = max_price_level

        places = []
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK":
                for place in data["results"]:
                    if exclude_place_id and place.get("place_id") == exclude_place_id:
                        continue

                    places.append({
                        "name": place.get("name"),
                        "place_id": place.get("place_id"),
                        "address": place.get("vicinity") or place.get("formatted_address"),
                        "latitude": place["geometry"]["location"]["lat"],
                        "longitude": place["geometry"]["location"]["lng"],
                        "rating": place.get("rating"),
                        "user_ratings_total": place.get("user_ratings_total"),
                        "types": place.get("types", []),
                        "price_level": place.get("price_level")
                    })
                return places
            elif data["status"] == "ZERO_RESULTS":
                return []
            else:
                return []

        except requests.exceptions.RequestException as e:
            return []
        except json.JSONDecodeError:
            return []

    def get_price_level_description(price_level):
        """Convert the price_level integer to a readable string."""
        if price_level is None:
            return "Not available"
        elif price_level == 0:
            return "Free"
        elif price_level == 1:
            return "Inexpensive"
        elif price_level == 2:
            return "Moderate"
        elif price_level == 3:
            return "Expensive"
        elif price_level == 4:
            return "Very Expensive"
        else:
            return "Unknown"

    # Main logic starts here
    # Extract locality from input place name for location bias
    parts = input_place_name.split(',')
    locality_name = parts[-1].strip() if len(parts) > 1 else None
    
    location_bias_coords = None
    if locality_name:
        lat_lng = get_coordinates_from_address(locality_name)
        if lat_lng:
            location_bias_coords = f"point:{lat_lng[0]},{lat_lng[1]}"

    # Get details of the input place
    initial_place_details = get_place_details_by_name(input_place_name, location_bias_coords)

    if not initial_place_details:
        return {
            'original_place': None,
            'nearby_places': [],
            'success': False,
            'error': f"Could not find details for '{input_place_name}'."
        }

    # Determine place types to search for
    place_types_to_search = []
    if 'restaurant' in initial_place_details['types']:
        place_types_to_search.append('restaurant')
    elif 'meal_takeaway' in initial_place_details['types']:
        place_types_to_search.append('meal_takeaway')
    elif 'food' in initial_place_details['types']:
        place_types_to_search.append('food')
    else:
        place_types_to_search.append('point_of_interest')

    # Set price range based on original place
    target_price_level = initial_place_details['price_level']
    min_p = max(0, target_price_level - 1) if target_price_level is not None else None
    max_p = min(4, target_price_level + 1) if target_price_level is not None else None

    # Find nearby places
    nearby_places = find_nearby_places_by_type_and_cost(
        latitude=initial_place_details['latitude'],
        longitude=initial_place_details['longitude'],
        search_types=place_types_to_search,
        radius=search_radius,
        min_price_level=min_p,
        max_price_level=max_p,
        exclude_place_id=initial_place_details['place_id']
    )

    return {
        'original_place': initial_place_details,
        'nearby_places': nearby_places,
        'success': True,
        'error': None
    }


# Example usage:
if __name__ == "__main__":
    result = find_similar_nearby_places("Manis Dum Biryani, Siruseri")
    
    # Access the returned data
    if result['success']:
        print(f"Found {len(result['nearby_places'])} similar places near {result['original_place']['name']}")
        for place in result['nearby_places']:
            print(f"- {place['name']} ({place['rating']} rating)")
    else:
        print(f"Error: {result['error']}")