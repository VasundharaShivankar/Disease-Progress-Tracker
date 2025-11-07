# TODO: Improve MongoDB Connection in upload_to_mongo.py

- [x] Update upload_to_mongo.py to use environment variables for CONNECTION_STRING and DATABASE_NAME
- [x] Add checks for missing environment variables and raise appropriate errors
- [x] Add error handling for MongoDB connection failures
- [x] Test the updated script (ensure env vars are set)
- [x] Verify connection in MongoDB Compass using the connection string

# TODO: Integrate Disease-Progress-Tracker into main nutritional_assessment repository

- [x] Add Disease-Progress-Tracker as submodule in ml-backend
- [x] Add "Disease Tracker" button in navigation bar
- [x] Change backend port to 5002 due to conflict on 5001
- [x] Ensure project opens with working existing models and disease progress tracker accessible via navbar
- [x] Add Psoriasis option to disease selector
- [x] Update frontend to use port 5002
- [x] Add fallback logic for Psoriasis in skin_analysis.py
- [x] Test backend endpoint for Psoriasis analysis
