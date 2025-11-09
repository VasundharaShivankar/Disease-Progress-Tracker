# TODO: Convert Tkinter app to Flask web app with navigation bar

- [x] Replace app.py with Flask application setup including routes for home (/), login (/login), progress-tracker (/progress-tracker), skin-analysis (/skin-analysis), spin-analysis (/spin-analysis)
- [x] Create templates/ directory
- [x] Create templates/index.html with Bootstrap-styled navigation bar containing buttons for login, home, progress-tracker, skin analysis, spin analysis
- [x] Create templates/login.html with basic login form placeholder
- [x] Create templates/progress_tracker.html with placeholder content
- [x] Create templates/skin_analysis.html with placeholder content
- [x] Create templates/spin_analysis.html with placeholder content
- [x] Create static/css/styles.css for custom styling (optional)
- [x] Install Flask if not already installed (pip install flask)
- [x] Run the Flask app (python app.py) and test navigation in browser
- [x] Verify that navigation buttons link correctly to routes
- [x] Integrate existing analysis logic into web forms (future enhancement)

# TODO: Convert to MERN stack

- [x] Create client/ React app
- [x] Create server/ Express app with basic routes
- [x] Install server dependencies: express, mongoose, cors, dotenv, bcryptjs, jsonwebtoken
- [x] Install client dependencies: axios, react-router-dom, bootstrap
- [x] Create Navbar component with navigation buttons
- [x] Create page components: Home, Login, ProgressTracker, SkinAnalysis, SpinAnalysis
- [x] Update App.js with routing
- [x] Start Flask app (running on http://127.0.0.1:5000)
- [ ] Start React client (need to fix npm start issue)
- [ ] Start Express server
- [ ] Test MERN stack navigation

# TODO: Integrate original Tkinter functionality into Progress Tracker

- [x] Add POST route for /progress-tracker with file upload handling
- [x] Integrate skin_analysis.py functions for image processing
- [x] Add disease type selection dropdown with same options as original app
- [x] Add image upload fields for past and new images
- [x] Generate analysis report with area calculations and progress status
- [x] Create matplotlib visualization showing before/after images and masks
- [x] Display results in web interface with Bootstrap styling
- [x] Add route to serve uploaded images (/uploads/<filename>)
- [x] Update progress_tracker.html template with form and results display
- [x] Test the complete progress tracking functionality

# TODO: Train Models Properly

- [x] Install required dependencies: tensorflow, opencv-python, matplotlib, pymongo
- [x] Run train_models.py to train segmentation models for all diseases
- [x] Verify models are saved in models/ directory
- [x] Test fallback segmentation methods for different disease types
- [x] Fix matplotlib backend issues for web server compatibility
- [x] Update app.py to use non-interactive matplotlib backend (Agg)
- [x] Test complete analysis pipeline with trained models
