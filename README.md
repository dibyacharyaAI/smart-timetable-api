
Smart Timetable API

This project provides a Flask-based API for generating and serving academic timetables using a trained autoencoder model.

---

##Project Structure

smart_timetable/
│
├── api_server.py # Main Flask server script
├── smart_timetable_system.py # Core logic for timetable generation & model handling
├── utils.py # Helper functions
├── requirements.txt # Python dependencies
├── pyproject.toml # Poetry/Build system config (optional)
├── uv.lock # Poetry lock file
├── smart_timetable_model.json # Pretrained model weights/config
├── data/
│ ├── activity_data_.csv
│ ├── teacher_data_.csv
│ ├── subject_data_.csv
│ ├── student_data_.csv
│ ├── transit_data.csv
│ ├── final_timetable_.csv
│ └── complete_transit_timetable_.csv


---

## Getting Started

### 1. Clone the Repository

```bash
📦 Setup Instructions
1. Clone the repo

git clone https://github.com/dibyacharyaAI/smart-timetable-api.git
cd smart_timetable-api
2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate
3. Install dependencies

pip install -r requirements.txt
4. Run the API

python api_server.py
🔌 API Endpoints
Base URL: http://localhost:5001/api

Method	Endpoint	Description
GET	/timetable	Full timetable
GET	/timetable/section/<SECTION_ID>	Timetable for a section (e.g. SEC01)
GET	/timetable/day/<DAY_INDEX>	Timetable for a day (0=Mon)
GET	/teacher/<TEACHER_ID>	Timetable for a teacher
GET	/status	Model & schedule status
POST	/regenerate	Trigger regeneration of timetable

 Health
GET /api/status
 Full Timetable
GET /api/timetable
 Timetable by Section
GET /api/timetable/section/<SECTION_ID>
 Timetable by Day
GET /api/timetable/day/<DAY_NUMBER>
 Teacher Timetable
GET /api/teacher/<TEACHER_ID>
 Regenerate Timetable
POST /api/regenerate
⚠ Notes

Streamlit-related warnings like missing ScriptRunContext are harmless in bare execution mode.
Avoid pushing venv/ or large .dylib files to GitHub – they exceed size limits.
 Maintainer



Dibyakanta Acharya


