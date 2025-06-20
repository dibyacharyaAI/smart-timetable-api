"""
Flask API Server for Smart Timetable System
Provides REST endpoints to fetch timetable data in specific JSON format
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_timetable_system import SmartTimetableSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global timetable system instance
timetable_system = None
current_schedule = None

def initialize_system():
    """Initialize the timetable system and load data"""
    global timetable_system, current_schedule
    
    try:
        timetable_system = SmartTimetableSystem()
        
        # Load all data files
        if timetable_system.load_all_data():
            print("✓ Data loaded successfully")
            
            # Generate schedule
            current_schedule = timetable_system.generate_complete_schedule()
            print(f"✓ Schedule generated with {len(current_schedule)} sections")
            
            return True
        else:
            print("✗ Failed to load data")
            return False
            
    except Exception as e:
        print(f"✗ Error initializing system: {str(e)}")
        return False

def get_day_number(time_slot):
    """Convert time slot to day number (1=Monday, 2=Tuesday, etc.)"""
    if not time_slot:
        return 1
    
    day_mapping = {
        'monday': 1, 'mon': 1,
        'tuesday': 2, 'tue': 2, 'tues': 2,
        'wednesday': 3, 'wed': 3,
        'thursday': 4, 'thu': 4, 'thur': 4, 'thurs': 4,
        'friday': 5, 'fri': 5,
        'saturday': 6, 'sat': 6,
        'sunday': 7, 'sun': 7
    }
    
    time_slot_lower = time_slot.lower()
    for day_name, day_num in day_mapping.items():
        if day_name in time_slot_lower:
            return day_num
    
    # Fallback: extract day from slot pattern like "Monday 9-10 AM"
    if 'monday' in time_slot_lower or 'mon' in time_slot_lower:
        return 1
    elif 'tuesday' in time_slot_lower or 'tue' in time_slot_lower:
        return 2
    elif 'wednesday' in time_slot_lower or 'wed' in time_slot_lower:
        return 3
    elif 'thursday' in time_slot_lower or 'thu' in time_slot_lower:
        return 4
    elif 'friday' in time_slot_lower or 'fri' in time_slot_lower:
        return 5
    
    return 1  # Default to Monday

def extract_time_hours(time_slot):
    """Extract start and end hours from time slot"""
    if not time_slot:
        return 9, 10  # Default
    
    import re
    
    # Look for patterns like "9-10", "09-10", "9:00-10:00", etc.
    time_patterns = [
        r'(\d{1,2})-(\d{1,2})',
        r'(\d{1,2}):00-(\d{1,2}):00',
        r'(\d{1,2})\.00-(\d{1,2})\.00'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, time_slot)
        if match:
            start_hour = int(match.group(1))
            end_hour = int(match.group(2))
            return start_hour, end_hour
    
    # Fallback: try to find individual numbers
    numbers = re.findall(r'\b(\d{1,2})\b', time_slot)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    
    return 9, 10  # Default fallback

def get_activity_color(activity_type):
    """Get color scheme for activity type"""
    color_mapping = {
        'THEORY': 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-100',
        'LAB': 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-100',
        'WORKSHOP': 'bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-100',
        'ELECTIVE': 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-100'
    }
    return color_mapping.get(activity_type, 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-100')

def format_schedule_entry(entry_id, section_id, time_slot, slot_data):
    """Format a single schedule entry to match required JSON structure"""
    
    # Extract day and time information
    day_num = get_day_number(time_slot)
    start_hour, end_hour = extract_time_hours(time_slot)
    
    # Get activity type and format it
    activity_type = slot_data.get('activity_type', 'THEORY')
    type_formatted = activity_type.lower()
    
    # Get subject and teacher info
    subject = slot_data.get('subject', 'Unknown Subject')
    teacher = slot_data.get('teacher', 'TBD')
    teacher_id = slot_data.get('teacher_id', 'T000')
    
    # Get room and campus info
    room = slot_data.get('room', 'Room-01')
    block_location = slot_data.get('block_location', 'Main Campus')
    
    # Format campus name
    campus = f"Campus {block_location.replace('-', ' ')}"
    
    return {
        "id": entry_id,
        "section": section_id,
        "title": subject,
        "day": day_num,
        "startHour": start_hour,
        "endHour": end_hour,
        "type": type_formatted,
        "room": room,
        "campus": campus,
        "teacher": teacher,
        "teacherId": teacher_id,
        "color": get_activity_color(activity_type)
    }

@app.route('/api/timetable', methods=['GET'])
def get_full_timetable():
    """Get complete timetable for all sections"""
    if not current_schedule:
        return jsonify({"error": "Schedule not generated yet"}), 500
    
    timetable_entries = []
    entry_id = 1
    
    for section_id, section_schedule in current_schedule.items():
        for time_slot, slot_data in section_schedule.items():
            entry = format_schedule_entry(entry_id, section_id, time_slot, slot_data)
            timetable_entries.append(entry)
            entry_id += 1
    
    return jsonify({
        "status": "success",
        "total_entries": len(timetable_entries),
        "timetable": timetable_entries
    })

@app.route('/api/timetable/section/<section_id>', methods=['GET'])
def get_section_timetable(section_id):
    """Get timetable for a specific section"""
    if not current_schedule:
        return jsonify({"error": "Schedule not generated yet"}), 500
    
    section_id_upper = section_id.upper()
    if section_id_upper not in current_schedule:
        return jsonify({"error": f"Section {section_id} not found"}), 404
    
    section_schedule = current_schedule[section_id_upper]
    timetable_entries = []
    entry_id = 1
    
    for time_slot, slot_data in section_schedule.items():
        entry = format_schedule_entry(entry_id, section_id_upper, time_slot, slot_data)
        timetable_entries.append(entry)
        entry_id += 1
    
    return jsonify({
        "status": "success",
        "section": section_id_upper,
        "total_entries": len(timetable_entries),
        "timetable": timetable_entries
    })

@app.route('/api/timetable/day/<int:day_num>', methods=['GET'])
def get_day_timetable(day_num):
    """Get timetable for a specific day (1=Monday, 2=Tuesday, etc.)"""
    if not current_schedule:
        return jsonify({"error": "Schedule not generated yet"}), 500
    
    if day_num < 1 or day_num > 7:
        return jsonify({"error": "Day number must be between 1-7"}), 400
    
    timetable_entries = []
    entry_id = 1
    
    for section_id, section_schedule in current_schedule.items():
        for time_slot, slot_data in section_schedule.items():
            if get_day_number(time_slot) == day_num:
                entry = format_schedule_entry(entry_id, section_id, time_slot, slot_data)
                timetable_entries.append(entry)
                entry_id += 1
    
    day_names = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    return jsonify({
        "status": "success",
        "day": day_names[day_num],
        "day_number": day_num,
        "total_entries": len(timetable_entries),
        "timetable": timetable_entries
    })

@app.route('/api/teacher/<teacher_id>', methods=['GET'])
def get_teacher_schedule(teacher_id):
    """Get schedule for a specific teacher"""
    if not current_schedule:
        return jsonify({"error": "Schedule not generated yet"}), 500
    
    timetable_entries = []
    entry_id = 1
    
    for section_id, section_schedule in current_schedule.items():
        for time_slot, slot_data in section_schedule.items():
            if slot_data.get('teacher_id') == teacher_id:
                entry = format_schedule_entry(entry_id, section_id, time_slot, slot_data)
                timetable_entries.append(entry)
                entry_id += 1
    
    return jsonify({
        "status": "success",
        "teacher_id": teacher_id,
        "total_entries": len(timetable_entries),
        "schedule": timetable_entries
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status and system information"""
    status_info = {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": timetable_system is not None,
        "schedule_available": current_schedule is not None,
        "total_sections": len(current_schedule) if current_schedule else 0
    }
    
    if current_schedule:
        total_entries = sum(len(section_schedule) for section_schedule in current_schedule.values())
        status_info["total_schedule_entries"] = total_entries
    
    return jsonify(status_info)

@app.route('/api/regenerate', methods=['POST'])
def regenerate_schedule():
    """Regenerate the timetable schedule"""
    global current_schedule
    
    if not timetable_system:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        current_schedule = timetable_system.generate_complete_schedule()
        total_entries = sum(len(section_schedule) for section_schedule in current_schedule.values())
        
        return jsonify({
            "status": "success",
            "message": "Schedule regenerated successfully",
            "total_sections": len(current_schedule),
            "total_entries": total_entries
        })
    except Exception as e:
        return jsonify({"error": f"Failed to regenerate schedule: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Smart Timetable API Server",
        "version": "1.0.0",
        "endpoints": {
            "/api/timetable": "Get complete timetable",
            "/api/timetable/section/<section_id>": "Get section timetable",
            "/api/timetable/day/<day_num>": "Get day timetable (1-7)",
            "/api/teacher/<teacher_id>": "Get teacher schedule",
            "/api/status": "Get API status",
            "/api/regenerate": "Regenerate schedule (POST)"
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Smart Timetable API Server...")
    
    # Initialize the system
    if initialize_system():
        print("✅ System initialized successfully")
        print("🌐 API Server running on http://0.0.0.0:5001")
        print("\nAvailable endpoints:")
        print("  GET  /api/timetable - Complete timetable")
        print("  GET  /api/timetable/section/SEC01 - Section timetable")
        print("  GET  /api/timetable/day/1 - Monday timetable")
        print("  GET  /api/teacher/T107 - Teacher schedule")
        print("  GET  /api/status - System status")
        print("  POST /api/regenerate - Regenerate schedule")
        
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("❌ Failed to initialize system. Please check data files.")