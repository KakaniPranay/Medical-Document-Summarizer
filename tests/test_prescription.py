import pytest

from prescription import build_diet_recommendations, build_educational_tablet_examples, build_precautions, build_prescription_pdf, build_prescription_text, extract_diagnosis_basis, extract_hospital_name, extract_json_object, extract_medications_from_text, extract_patient_details, extract_report_signals, normalize_payload 
 
def test_extract_json_object_from_fenced_block(): 
    raw = '```json\n{\"patient_summary\": \"Stable\"}\n```' 
    assert extract_json_object(raw) == '{\"patient_summary\": \"Stable\"}' 
 
def test_normalize_payload_adds_defaults_and_formats_text(): 
    payload = normalize_payload({'patient_summary': 'Fever and cough', 'medications': ['Paracetamol']}, 'Fallback summary') 
    assert payload['patient_summary'] == 'Fever and cough' 
    assert payload['medications'][0]['name'] == 'Paracetamol' 
    assert payload['missing_information'] 
    assert payload['precautions']
    text_output = build_prescription_text(payload) 
    assert 'Draft prescription for clinician review:' in text_output
    assert 'Precautions:' in text_output

def test_extract_hospital_name_prefers_known_list_match():
    report = 'Discharge Summary Apollo Hospitals Name: Jane Doe Age: 45 years'
    assert extract_hospital_name(report) == 'Apollo Hospitals'

def test_report_signals_drive_diet_recommendations():
    signals = extract_report_signals('Patient has diabetes and hypertension. HbA1c and fasting blood sugar were elevated.')
    advice = build_diet_recommendations(signals)
    assert 'diabetes' in signals['conditions']
    assert 'blood sugar tests' in signals['tests']
    assert any('sugar' in item.lower() or 'salt' in item.lower() for item in advice)

def test_extract_medications_and_diagnosis_from_report_text():
    report = (
        'Diagnosis: Type 2 diabetes mellitus with hypertension\n'
        'Medications:\n'
        'Tab Metformin 500 mg BD after food for 30 days\n'
        'Tab Amlodipine 5 mg OD\n'
    )
    signals = extract_report_signals(report)
    diagnosis = extract_diagnosis_basis(report, signals)
    medications = extract_medications_from_text(report, signals)

    assert any('diabetes' in item.lower() for item in diagnosis)
    assert any(item['name'].lower().startswith('metformin') for item in medications)
    assert any(item['frequency'] in ['BD', 'OD'] for item in medications)
    assert any('after food' in item['instructions'].lower() for item in medications)

def test_extract_patient_details_prefers_labeled_fields():
    report = 'Patient Name: John Doe Age: 45 years Gender: Male UHID: AB1234 Consultant: Dr Ravi Kumar'
    details = extract_patient_details(report)
    assert details['name'] == 'John Doe'
    assert '45' in details['age']
    assert details['sex'].lower() == 'male'
    assert details['patient_id'] == 'AB1234'

def test_build_precautions_uses_conditions_tests_and_meal_timing():
    signals = {
        'conditions': ['diabetes', 'kidney disease'],
        'tests': ['blood sugar tests', 'kidney function tests'],
    }
    medications = [
        {
            'name': 'Metformin',
            'instructions': 'Take after food.',
        }
    ]

    precautions = build_precautions(signals, medications)

    assert any('blood sugar tests' in item.lower() for item in precautions)
    assert any('renal dosing' in item.lower() for item in precautions)
    assert any('meal-related instructions' in item.lower() for item in precautions)

def test_build_educational_tablet_examples_are_condition_based():
    examples = build_educational_tablet_examples(
        {'conditions': ['diabetes', 'hypertension'], 'tests': []}
    )

    names = [item['name'].lower() for item in examples]

    assert 'metformin' in names
    assert 'amlodipine' in names
    assert all('educational' in item['warning'].lower() for item in examples)

def test_build_prescription_pdf_returns_pdf_bytes():
    pytest.importorskip('reportlab')
    payload = normalize_payload(
        {
            'patient_summary': 'Type 2 diabetes with elevated blood sugar.',
            'diagnosis_basis': ['Type 2 diabetes mellitus'],
            'medications': [
                {
                    'name': 'Metformin',
                    'dosage': '500 mg',
                    'frequency': 'BD',
                    'duration': 'for 30 days',
                    'route': 'oral',
                    'instructions': 'Take after food.',
                    'status': 'report-derived',
                }
            ],
        },
        'Fallback summary',
    )

    pdf_bytes = build_prescription_pdf(payload)

    assert pdf_bytes.startswith(b'%PDF')
