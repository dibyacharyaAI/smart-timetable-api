import pandas as pd
import numpy as np
import csv
import io
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import traceback

class Utils:
    """Utility functions for timetable operations"""
    
    @staticmethod
    def convert_timetable_to_csv(timetable_data: Dict[str, Any]) -> str:
        """Convert timetable data to CSV format"""
        try:
            schedule = timetable_data.get('schedule', {})
            
            # Convert to list of dictionaries
            csv_rows = []
            
            for section_id, section_schedule in schedule.items():
                for time_slot, slot_data in section_schedule.items():
                    # Parse time slot
                    day, time = time_slot.split('_', 1)
                    time_display = time.replace('_', ':')
                    
                    row = {
                        'Section': section_id,
                        'Day': day,
                        'Time': time_display,
                        'TimeSlot': time_slot,
                        'Subject': slot_data.get('subject', ''),
                        'SubjectName': slot_data.get('subject_name', ''),
                        'Teacher': slot_data.get('teacher', ''),
                        'Room': slot_data.get('room', ''),
                        'RoomType': slot_data.get('room_type', ''),
                        'ActivityType': slot_data.get('activity_type', ''),
                        'Scheme': slot_data.get('scheme', '')
                    }
                    
                    csv_rows.append(row)
            
            # Convert to CSV string
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                return df.to_csv(index=False)
            else:
                return "Section,Day,Time,TimeSlot,Subject,SubjectName,Teacher,Room,RoomType,ActivityType,Scheme\n"
                
        except Exception as e:
            raise Exception(f"Error converting timetable to CSV: {str(e)}")
    
    @staticmethod
    def convert_csv_to_timetable(csv_content: str) -> Dict[str, Any]:
        """Convert CSV content back to timetable format"""
        try:
            # Read CSV content
            csv_file = io.StringIO(csv_content)
            df = pd.read_csv(csv_file)
            
            # Rebuild schedule structure
            schedule = {}
            
            for _, row in df.iterrows():
                section_id = row['Section']
                time_slot = row['TimeSlot']
                
                if section_id not in schedule:
                    schedule[section_id] = {}
                
                slot_data = {
                    'subject': row.get('Subject', ''),
                    'subject_name': row.get('SubjectName', ''),
                    'teacher': row.get('Teacher', ''),
                    'room': row.get('Room', ''),
                    'room_type': row.get('RoomType', ''),
                    'activity_type': row.get('ActivityType', ''),
                    'scheme': row.get('Scheme', '')
                }
                
                schedule[section_id][time_slot] = slot_data
            
            return {
                'schedule': schedule,
                'metadata': {
                    'import_timestamp': datetime.now().isoformat(),
                    'total_sections': len(schedule),
                    'total_slots': sum(len(slots) for slots in schedule.values())
                }
            }
            
        except Exception as e:
            raise Exception(f"Error converting CSV to timetable: {str(e)}")
    
    @staticmethod
    def export_teacher_schedule_csv(timetable_data: Dict[str, Any]) -> str:
        """Export teacher-wise schedule as CSV"""
        try:
            schedule = timetable_data.get('schedule', {})
            
            # Build teacher schedule
            teacher_rows = []
            
            for section_id, section_schedule in schedule.items():
                for time_slot, slot_data in section_schedule.items():
                    teacher = slot_data.get('teacher')
                    if teacher and teacher != 'TBD':
                        day, time = time_slot.split('_', 1)
                        time_display = time.replace('_', ':')
                        
                        row = {
                            'Teacher': teacher,
                            'Day': day,
                            'Time': time_display,
                            'Section': section_id,
                            'Subject': slot_data.get('subject_name', slot_data.get('subject', '')),
                            'Room': slot_data.get('room', ''),
                            'RoomType': slot_data.get('room_type', ''),
                            'ActivityType': slot_data.get('activity_type', '')
                        }
                        
                        teacher_rows.append(row)
            
            if teacher_rows:
                df = pd.DataFrame(teacher_rows)
                df = df.sort_values(['Teacher', 'Day', 'Time'])
                return df.to_csv(index=False)
            else:
                return "Teacher,Day,Time,Section,Subject,Room,RoomType,ActivityType\n"
                
        except Exception as e:
            raise Exception(f"Error exporting teacher schedule: {str(e)}")
    
    @staticmethod
    def export_room_utilization_csv(timetable_data: Dict[str, Any]) -> str:
        """Export room utilization as CSV"""
        try:
            schedule = timetable_data.get('schedule', {})
            
            # Build room utilization data
            room_rows = []
            
            for section_id, section_schedule in schedule.items():
                for time_slot, slot_data in section_schedule.items():
                    room = slot_data.get('room')
                    if room and room != 'TBD':
                        day, time = time_slot.split('_', 1)
                        time_display = time.replace('_', ':')
                        
                        row = {
                            'Room': room,
                            'RoomType': slot_data.get('room_type', ''),
                            'Day': day,
                            'Time': time_display,
                            'Section': section_id,
                            'Subject': slot_data.get('subject_name', slot_data.get('subject', '')),
                            'Teacher': slot_data.get('teacher', ''),
                            'ActivityType': slot_data.get('activity_type', '')
                        }
                        
                        room_rows.append(row)
            
            if room_rows:
                df = pd.DataFrame(room_rows)
                df = df.sort_values(['Room', 'Day', 'Time'])
                return df.to_csv(index=False)
            else:
                return "Room,RoomType,Day,Time,Section,Subject,Teacher,ActivityType\n"
                
        except Exception as e:
            raise Exception(f"Error exporting room utilization: {str(e)}")
    
    @staticmethod
    def validate_csv_format(csv_content: str) -> Dict[str, Any]:
        """Validate CSV format and return validation results"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'row_count': 0,
                'column_info': {}
            }
            
            # Read CSV
            csv_file = io.StringIO(csv_content)
            df = pd.read_csv(csv_file)
            
            validation_result['row_count'] = len(df)
            
            # Check required columns
            required_columns = ['Section', 'Day', 'Time', 'TimeSlot', 'Subject', 'Teacher', 'Room']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Check for empty values
            for col in required_columns:
                if col in df.columns:
                    empty_count = df[col].isna().sum() + (df[col] == '').sum()
                    if empty_count > 0:
                        validation_result['warnings'].append(f"Column '{col}' has {empty_count} empty values")
            
            # Validate time format
            if 'Time' in df.columns:
                invalid_times = []
                for idx, time_val in enumerate(df['Time']):
                    if pd.notna(time_val):
                        time_str = str(time_val)
                        if ':' not in time_str or len(time_str.split(':')) != 2:
                            invalid_times.append(idx + 1)
                
                if invalid_times:
                    validation_result['warnings'].append(f"Invalid time format in rows: {invalid_times[:5]}...")
            
            # Validate days
            if 'Day' in df.columns:
                valid_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
                invalid_days = df[~df['Day'].isin(valid_days + [np.nan])]['Day'].unique()
                
                if len(invalid_days) > 0:
                    validation_result['warnings'].append(f"Invalid day values: {list(invalid_days)}")
            
            # Column information
            validation_result['column_info'] = {
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict()
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"CSV validation error: {str(e)}"],
                'warnings': [],
                'row_count': 0,
                'column_info': {}
            }
    
    @staticmethod
    def generate_statistics_summary(timetable_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics summary"""
        try:
            schedule = timetable_data.get('schedule', {})
            summary = timetable_data.get('summary', {})
            
            stats = {
                'overview': {
                    'total_sections': len(schedule),
                    'total_scheduled_slots': sum(len(slots) for slots in schedule.values()),
                    'generation_timestamp': datetime.now().isoformat()
                },
                'distribution': {
                    'subjects': {},
                    'teachers': {},
                    'rooms': {},
                    'days': {'Mon': 0, 'Tue': 0, 'Wed': 0, 'Thu': 0, 'Fri': 0},
                    'times': {}
                },
                'utilization': {
                    'teacher_workload': {},
                    'room_occupancy': {},
                    'section_density': {}
                },
                'quality_metrics': {
                    'elective_coverage': 0,
                    'schedule_density': 0,
                    'teacher_efficiency': 0
                }
            }
            
            # Calculate distributions
            for section_id, section_schedule in schedule.items():
                section_slots = len(section_schedule)
                stats['utilization']['section_density'][section_id] = section_slots
                
                for time_slot, slot_data in section_schedule.items():
                    # Day distribution
                    day = time_slot.split('_')[0]
                    if day in stats['distribution']['days']:
                        stats['distribution']['days'][day] += 1
                    
                    # Time distribution
                    time_part = time_slot.split('_', 1)[1]
                    stats['distribution']['times'][time_part] = stats['distribution']['times'].get(time_part, 0) + 1
                    
                    # Subject distribution
                    subject = slot_data.get('subject', '')
                    stats['distribution']['subjects'][subject] = stats['distribution']['subjects'].get(subject, 0) + 1
                    
                    # Teacher distribution and workload
                    teacher = slot_data.get('teacher', '')
                    if teacher and teacher != 'TBD':
                        stats['distribution']['teachers'][teacher] = stats['distribution']['teachers'].get(teacher, 0) + 1
                        stats['utilization']['teacher_workload'][teacher] = stats['utilization']['teacher_workload'].get(teacher, 0) + 1
                    
                    # Room distribution and occupancy
                    room = slot_data.get('room', '')
                    if room and room != 'TBD':
                        stats['distribution']['rooms'][room] = stats['distribution']['rooms'].get(room, 0) + 1
                        stats['utilization']['room_occupancy'][room] = stats['utilization']['room_occupancy'].get(room, 0) + 1
            
            # Calculate quality metrics
            total_possible_slots = len(schedule) * 30  # 30 slots per week per section
            if total_possible_slots > 0:
                stats['quality_metrics']['schedule_density'] = (stats['overview']['total_scheduled_slots'] / total_possible_slots) * 100
            
            # Elective coverage
            elective_slots = stats['distribution']['subjects'].get('ELECTIVE', 0)
            expected_electives = len(schedule) * 5  # 5 days per week
            if expected_electives > 0:
                stats['quality_metrics']['elective_coverage'] = (elective_slots / expected_electives) * 100
            
            # Teacher efficiency (average classes per teacher)
            if stats['distribution']['teachers']:
                avg_classes = sum(stats['utilization']['teacher_workload'].values()) / len(stats['utilization']['teacher_workload'])
                stats['quality_metrics']['teacher_efficiency'] = avg_classes
            
            return stats
            
        except Exception as e:
            return {'error': f"Error generating statistics: {str(e)}"}
    
    @staticmethod
    def format_time_slot(time_slot: str) -> str:
        """Format time slot for display"""
        try:
            day, time = time_slot.split('_', 1)
            time_display = time.replace('_', ':')
            
            # Convert to 12-hour format
            hour, minute = time_display.split(':')
            hour_int = int(hour)
            
            if hour_int == 12:
                period = 'PM'
            elif hour_int > 12:
                hour_int -= 12
                period = 'PM'
            else:
                period = 'AM'
                if hour_int == 0:
                    hour_int = 12
            
            formatted_time = f"{hour_int:02d}:{minute} {period}"
            
            # Day abbreviations
            day_names = {
                'Mon': 'Monday',
                'Tue': 'Tuesday', 
                'Wed': 'Wednesday',
                'Thu': 'Thursday',
                'Fri': 'Friday'
            }
            
            return f"{day_names.get(day, day)} {formatted_time}"
            
        except Exception:
            return time_slot
    
    @staticmethod
    def parse_time_slot(formatted_time: str) -> str:
        """Parse formatted time back to time slot format"""
        try:
            # This is a simplified parser - might need enhancement
            parts = formatted_time.split()
            if len(parts) >= 2:
                day_name = parts[0]
                time_part = parts[1]
                
                # Convert day name to abbreviation
                day_abbr = {
                    'Monday': 'Mon',
                    'Tuesday': 'Tue',
                    'Wednesday': 'Wed', 
                    'Thursday': 'Thu',
                    'Friday': 'Fri'
                }
                
                day = day_abbr.get(day_name, day_name[:3])
                
                # Convert time to slot format (simplified)
                time_slot_format = time_part.replace(':', '_').replace(' AM', '').replace(' PM', '')
                
                return f"{day}_{time_slot_format}"
            
            return formatted_time
            
        except Exception:
            return formatted_time
    
    @staticmethod
    def export_json(timetable_data: Dict[str, Any]) -> str:
        """Export timetable data as JSON"""
        try:
            # Make data JSON serializable
            json_data = Utils._make_json_serializable(timetable_data)
            return json.dumps(json_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            raise Exception(f"Error exporting JSON: {str(e)}")
    
    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: Utils._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Utils._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    @staticmethod
    def calculate_workload_balance(timetable_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workload balance metrics"""
        try:
            schedule = timetable_data.get('schedule', {})
            
            # Teacher workload
            teacher_workload = {}
            
            for section_schedule in schedule.values():
                for slot_data in section_schedule.values():
                    teacher = slot_data.get('teacher')
                    if teacher and teacher != 'TBD':
                        teacher_workload[teacher] = teacher_workload.get(teacher, 0) + 1
            
            if not teacher_workload:
                return {'error': 'No teacher workload data available'}
            
            workloads = list(teacher_workload.values())
            
            balance_metrics = {
                'total_teachers': len(teacher_workload),
                'total_classes': sum(workloads),
                'average_workload': np.mean(workloads),
                'median_workload': np.median(workloads),
                'min_workload': min(workloads),
                'max_workload': max(workloads),
                'workload_std': np.std(workloads),
                'balance_coefficient': np.std(workloads) / np.mean(workloads) if np.mean(workloads) > 0 else 0,
                'teacher_workloads': teacher_workload
            }
            
            # Categorize teachers by workload
            avg_workload = balance_metrics['average_workload']
            
            balance_metrics['workload_categories'] = {
                'underloaded': [t for t, w in teacher_workload.items() if w < avg_workload * 0.7],
                'balanced': [t for t, w in teacher_workload.items() if avg_workload * 0.7 <= w <= avg_workload * 1.3],
                'overloaded': [t for t, w in teacher_workload.items() if w > avg_workload * 1.3]
            }
            
            return balance_metrics
            
        except Exception as e:
            return {'error': f"Error calculating workload balance: {str(e)}"}
    
    @staticmethod
    def generate_conflict_report(timetable_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed conflict report"""
        try:
            schedule = timetable_data.get('schedule', {})
            
            conflicts = {
                'teacher_conflicts': [],
                'room_conflicts': [],
                'summary': {
                    'total_conflicts': 0,
                    'conflict_free': True
                }
            }
            
            # Check teacher conflicts
            teacher_slots = {}
            room_slots = {}
            
            for section_id, section_schedule in schedule.items():
                for time_slot, slot_data in section_schedule.items():
                    teacher = slot_data.get('teacher')
                    room = slot_data.get('room')
                    
                    # Track teacher assignments
                    if teacher and teacher != 'TBD':
                        if time_slot not in teacher_slots:
                            teacher_slots[time_slot] = {}
                        if teacher not in teacher_slots[time_slot]:
                            teacher_slots[time_slot][teacher] = []
                        teacher_slots[time_slot][teacher].append({
                            'section': section_id,
                            'subject': slot_data.get('subject', ''),
                            'room': room
                        })
                    
                    # Track room assignments
                    if room and room != 'TBD':
                        if time_slot not in room_slots:
                            room_slots[time_slot] = {}
                        if room not in room_slots[time_slot]:
                            room_slots[time_slot][room] = []
                        room_slots[time_slot][room].append({
                            'section': section_id,
                            'subject': slot_data.get('subject', ''),
                            'teacher': teacher
                        })
            
            # Find teacher conflicts
            for time_slot, teachers in teacher_slots.items():
                for teacher, assignments in teachers.items():
                    if len(assignments) > 1:
                        conflicts['teacher_conflicts'].append({
                            'teacher': teacher,
                            'time_slot': time_slot,
                            'conflicting_assignments': assignments
                        })
            
            # Find room conflicts  
            for time_slot, rooms in room_slots.items():
                for room, assignments in rooms.items():
                    if len(assignments) > 1:
                        conflicts['room_conflicts'].append({
                            'room': room,
                            'time_slot': time_slot,
                            'conflicting_assignments': assignments
                        })
            
            # Update summary
            total_conflicts = len(conflicts['teacher_conflicts']) + len(conflicts['room_conflicts'])
            conflicts['summary']['total_conflicts'] = total_conflicts
            conflicts['summary']['conflict_free'] = total_conflicts == 0
            conflicts['summary']['teacher_conflict_count'] = len(conflicts['teacher_conflicts'])
            conflicts['summary']['room_conflict_count'] = len(conflicts['room_conflicts'])
            
            return conflicts
            
        except Exception as e:
            return {'error': f"Error generating conflict report: {str(e)}"}
    
    @staticmethod
    def backup_timetable(timetable_data: Dict[str, Any], backup_name: str = None) -> str:
        """Create backup of timetable data"""
        try:
            if backup_name is None:
                backup_name = f"timetable_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create backup data
            backup_data = {
                'backup_timestamp': datetime.now().isoformat(),
                'backup_name': backup_name,
                'timetable_data': timetable_data
            }
            
            # Convert to JSON
            json_backup = Utils.export_json(backup_data)
            
            return json_backup
            
        except Exception as e:
            raise Exception(f"Error creating backup: {str(e)}")
    
    @staticmethod
    def restore_from_backup(backup_json: str) -> Dict[str, Any]:
        """Restore timetable from backup"""
        try:
            backup_data = json.loads(backup_json)
            
            if 'timetable_data' not in backup_data:
                raise ValueError("Invalid backup format: missing timetable_data")
            
            return backup_data['timetable_data']
            
        except Exception as e:
            raise Exception(f"Error restoring from backup: {str(e)}")
