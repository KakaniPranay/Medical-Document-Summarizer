import io
import json 
import logging 
import re
import textwrap
 
logger = logging.getLogger(__name__) 
 
DEFAULT_MISSING_INFORMATION = [ 
    'Confirmed working diagnosis', 
    'Patient age and weight', 
    'Drug allergies and prior adverse reactions', 
    'Current medications and interaction risks', 
    'Renal and hepatic function', 
    'Pregnancy or breastfeeding status if relevant', 
] 
 
DEFAULT_WARNINGS = [ 
    'Draft for clinician review only. Do not use as a final prescription.', 
    'Verify allergies, contraindications, interactions, and recent clinical status before prescribing.', 
] 

DIET_GUIDANCE_RULES = {
    'diabetes': [
        'Prefer vegetables, pulses, and high-fiber foods with controlled portions of rice, bread, or other carbohydrates.',
        'Limit sugary drinks, sweets, and frequent refined snacks.',
        'Try to keep meal timing regular to help control blood sugar.',
    ],
    'hypertension': [
        'Reduce salt intake and avoid heavily processed foods.',
        'Choose fruits, vegetables, pulses, and lighter home-cooked meals more often.',
    ],
    'kidney disease': [
        'Limit excess salt and follow any fluid, potassium, or protein advice given after kidney test review.',
        'Avoid starting supplements without clinician review.',
    ],
    'liver disease': [
        'Avoid alcohol completely unless the treating clinician has clearly advised otherwise.',
        'Prefer balanced meals and avoid foods that worsen nausea or bloating.',
    ],
    'anemia': [
        'Include iron-rich foods such as leafy greens, beans, lentils, and iron-fortified foods.',
        'Take vitamin C-rich foods with meals when possible to support iron absorption.',
    ],
    'high cholesterol': [
        'Reduce fried foods, excess butter, and packaged snacks high in trans fats.',
        'Prefer oats, vegetables, and lean protein sources regularly.',
    ],
    'thyroid disorder': [
        'Take thyroid medicines at the same time each day as advised.',
        'Avoid changing iodine supplement use unless the clinician recommends it.',
    ],
    'gastritis': [
        'Avoid foods that trigger burning or acidity, especially very spicy, oily, or late-night meals.',
        'Prefer smaller, simpler meals if heavy meals worsen symptoms.',
    ],
}

EDUCATIONAL_TABLET_OPTIONS = {
    'diabetes': [
        {'name': 'Metformin', 'use': 'Commonly used for type 2 diabetes', 'warning': 'Educational example only. Final medicine choice depends on kidney function, tolerance, and clinician review.'},
    ],
    'hypertension': [
        {'name': 'Amlodipine', 'use': 'Commonly used for high blood pressure', 'warning': 'Educational example only. Blood pressure pattern, swelling risk, and clinician review matter.'},
    ],
    'high cholesterol': [
        {'name': 'Atorvastatin', 'use': 'Commonly used to lower cholesterol', 'warning': 'Educational example only. Liver history, interactions, and clinician review matter.'},
    ],
    'thyroid disorder': [
        {'name': 'Levothyroxine', 'use': 'Commonly used for hypothyroidism', 'warning': 'Educational example only. Dose selection depends on thyroid tests and clinician review.'},
    ],
    'gastritis': [
        {'name': 'Pantoprazole', 'use': 'Commonly used to reduce stomach acid symptoms', 'warning': 'Educational example only. The underlying cause still needs clinician assessment.'},
    ],
    'anemia': [
        {'name': 'Ferrous sulfate', 'use': 'Commonly used for iron deficiency anemia', 'warning': 'Educational example only. The cause of anemia should be confirmed before treatment.'},
    ],
}

TEST_KEYWORDS = {
    'blood sugar tests': ['hba1c', 'fasting blood sugar', 'fbs', 'ppbs', 'rbs', 'glucose'],
    'kidney function tests': ['creatinine', 'urea', 'egfr', 'bun'],
    'liver function tests': ['sgot', 'sgpt', 'ast', 'alt', 'bilirubin', 'lft'],
    'lipid profile': ['cholesterol', 'ldl', 'hdl', 'triglyceride', 'lipid'],
    'blood count': ['hemoglobin', 'haemoglobin', 'cbc', 'rbc', 'wbc', 'platelet'],
    'thyroid profile': ['tsh', 't3', 't4', 'thyroid'],
}

CONDITION_KEYWORDS = {
    'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'high blood sugar'],
    'hypertension': ['hypertension', 'high blood pressure'],
    'kidney disease': ['ckd', 'kidney disease', 'renal failure', 'creatinine'],
    'liver disease': ['hepatitis', 'fatty liver', 'cirrhosis', 'liver disease', 'bilirubin'],
    'anemia': ['anemia', 'anaemia', 'low hemoglobin', 'low haemoglobin'],
    'high cholesterol': ['hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'ldl'],
    'thyroid disorder': ['hypothyroid', 'hyperthyroid', 'thyroid'],
    'gastritis': ['gastritis', 'acid peptic disease', 'reflux', 'gerd', 'acidity'],
}

MEDICATION_SECTION_HEADERS = [
    'medications',
    'medication',
    'prescription',
    'rx',
    'treatment advised',
    'treatment',
    'discharge medications',
]

DOSAGE_UNITS = ['mg', 'mcg', 'g', 'ml', 'units']
FREQUENCY_HINTS = [
    'od', 'bd', 'tds', 'qid', 'hs', 'stat',
    'once daily', 'twice daily', 'thrice daily',
    'every day', 'every morning', 'every night',
    'before food', 'after food', 'with food',
]

DIAGNOSIS_HEADERS = ['diagnosis', 'impression', 'assessment', 'clinical impression', 'provisional diagnosis']

PATIENT_DETAIL_FIELDS = [
    'name',
    'age',
    'sex',
    'dob',
    'patient_id',
    'doctor',
]

PATIENT_DETAIL_LABELS = {
    'name': 'Name',
    'age': 'Age',
    'sex': 'Sex',
    'dob': 'DOB',
    'patient_id': 'Patient ID',
    'doctor': 'Doctor',
}

PATIENT_DETAIL_PATTERNS = {
    'name': [
        r'(?im)\b(?:patient\s+name|patient)\s*[:\-]\s*([A-Za-z .]{2,60}?)(?=\s+(?:age|sex|gender|dob|patient\s+id|mrn|uhid|doctor|consultant)\b|$)',
        r'(?im)\bname\s*[:\-]\s*([A-Za-z .]{2,60}?)(?=\s+(?:age|sex|gender|dob|patient\s+id|mrn|uhid|doctor|consultant)\b|$)',
    ],
    'age': [
        r'(?im)\bage\s*[:\-]\s*([^\n,]{1,20}?)(?=\s+(?:sex|gender|dob|patient\s+id|mrn|uhid|doctor|consultant)\b|$)',
        r'(?im)\b(\d{1,3}\s*(?:years?|yrs?)\s*(?:old)?)\b',
    ],
    'sex': [
        r'(?im)\b(?:sex|gender)\s*[:\-]\s*([^\n,]{1,20}?)(?=\s+(?:dob|patient\s+id|mrn|uhid|doctor|consultant)\b|$)',
    ],
    'dob': [
        r'(?im)\b(?:dob|date\s+of\s+birth)\s*[:\-]\s*([^\n,]{1,30}?)(?=\s+(?:patient\s+id|mrn|uhid|doctor|consultant)\b|$)',
    ],
    'patient_id': [
        r'(?im)\b(?:patient\s+id|mrn|uhid|ip\s*no|op\s*no|registration\s*no)\s*[:\-]\s*([A-Za-z0-9\-/]+)',
    ],
    'doctor': [
        r'(?im)\b(?:doctor|consultant|physician)\s*[:\-]\s*([A-Za-z .]{2,60})',
        r'(?im)\b(?:dr\.?|dr)\s+([A-Za-z .]{2,60})',
    ],
}

HOSPITAL_PATTERNS = [
    r'(?im)^\s*(?:hospital|clinic|medical\s+center|medical\s+centre|healthcare\s+center|healthcare\s+centre)\s*[:\-]\s*([^\n]+)',
]

KNOWN_HOSPITAL_ALIASES = {
    'Apollo Hospitals': ['apollo hospitals', 'apollo hospital'],
    'Fortis Hospital': ['fortis hospital', 'fortis hospitals', 'fortis healthcare'],
    'Max Super Speciality Hospital': ['max super speciality hospital', 'max hospital', 'max healthcare'],
    'Manipal Hospital': ['manipal hospital', 'manipal hospitals'],
    'Narayana Health': ['narayana health', 'narayana hospital', 'narayana hrudayalaya'],
    'Medanta': ['medanta', 'medanta the medicity'],
    'AIIMS': ['aiims', 'all india institute of medical sciences'],
    'BLK-Max Super Speciality Hospital': ['blk max super speciality hospital', 'blk hospital'],
    'Sir Ganga Ram Hospital': ['sir ganga ram hospital'],
    'Artemis Hospital': ['artemis hospital'],
    'Aster CMI Hospital': ['aster cmi hospital', 'aster hospital'],
    'CARE Hospitals': ['care hospitals', 'care hospital'],
    'Yashoda Hospitals': ['yashoda hospitals', 'yashoda hospital'],
    'JSPS Govt Homeopathic Hospital': ['jsps govt homeopathic hospital', 'jsps homeopathic hospital', 'jsps hospital', 'jsps'],
    'KIMS Hospital': ['kims hospital', 'kims hospitals'],
    'Global Hospitals': ['global hospitals', 'global hospital'],
    'MGM Healthcare': ['mgm healthcare', 'mgm hospital'],
    'Kokilaben Dhirubhai Ambani Hospital': ['kokilaben dhirubhai ambani hospital', 'kokilaben hospital'],
    'Lilavati Hospital': ['lilavati hospital'],
    'Tata Memorial Hospital': ['tata memorial hospital'],
    'Christian Medical College': ['christian medical college', 'cmc vellore', 'cmc hospital'],
    'Mayo Clinic': ['mayo clinic'],
    'Cleveland Clinic': ['cleveland clinic'],
    'Johns Hopkins Hospital': ['johns hopkins hospital', 'johns hopkins'],
    'Massachusetts General Hospital': ['massachusetts general hospital', 'mgh hospital', 'mgh'],
    'NewYork-Presbyterian Hospital': ['newyork presbyterian hospital', 'new york presbyterian hospital'],
    'Mount Sinai Hospital': ['mount sinai hospital', 'mount sinai'],
    'Cedars-Sinai Medical Center': ['cedars sinai medical center', 'cedars sinai'],
    'Kaiser Permanente': ['kaiser permanente'],
}
 
def extract_json_object(text): 
    text = (text or '').strip() 
    if text.startswith('```'): 
        lines = text.splitlines() 
        if lines: 
            lines = lines[1:] 
        if lines and lines[-1].strip() == '```': 
            lines = lines[:-1] 
        text = '\n'.join(lines).strip() 
    start = text.find('{') 
    end = text.rfind('}') 
    if start != -1 and end != -1 and end + 1 > start: 
        return text[start:end + 1] 
    return text or '{}'
 
def normalize_list(value): 
    if value is None: 
        return [] 
    if isinstance(value, str): 
        items = [value] 
    elif isinstance(value, list): 
        items = value 
    else: 
        items = [str(value)] 
    cleaned = [] 
    for item in items: 
        text = str(item).strip() 
        if text and text not in cleaned: 
            cleaned.append(text) 
    return cleaned 

def clean_extracted_value(value):
    text = str(value or '').strip().strip('.,;:-')
    text = re.sub(r'\s+', ' ', text)
    return text

def shorten_patient_summary(text, sentence_limit=2):
    cleaned = clean_extracted_value(text)
    if not cleaned:
        return 'Insufficient information'
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    shortened = ' '.join(sentences[:sentence_limit]).strip()
    return shortened or cleaned

def normalize_match_text(value):
    text = str(value or '').lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_patient_details(value, fallback=None):
    fallback = fallback or {}
    details = {}
    if isinstance(value, dict):
        for field in PATIENT_DETAIL_FIELDS:
            details[field] = clean_extracted_value(value.get(field, ''))
    for field in PATIENT_DETAIL_FIELDS:
        if not details.get(field):
            details[field] = clean_extracted_value(fallback.get(field, ''))
        if not details.get(field):
            details[field] = 'Not provided'
    return details

def _valid_patient_detail(field, value):
    value = clean_extracted_value(value)
    lowered = value.lower()
    if not value:
        return False
    if field == 'name':
        if any(token in lowered for token in ['hospital', 'diagnosis', 'impression', 'medication', 'tablet', 'capsule']):
            return False
        return bool(re.fullmatch(r'[A-Za-z .]{2,60}', value)) and len(value.split()) <= 5
    if field == 'age':
        return bool(re.search(r'\d', value))
    if field == 'sex':
        return lowered in ['male', 'female', 'm', 'f', 'other']
    if field == 'dob':
        return bool(re.search(r'\d', value))
    if field == 'patient_id':
        return len(value) <= 25
    if field == 'doctor':
        return bool(re.fullmatch(r'[A-Za-z .]{2,60}', value))
    return True

def extract_patient_details(text):
    text = text or ''
    details = {}
    for field, patterns in PATIENT_DETAIL_PATTERNS.items():
        extracted = ''
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = clean_extracted_value(match.group(1))
                if _valid_patient_detail(field, candidate):
                    extracted = candidate
                    break
        details[field] = extracted or 'Not provided'
    return details

def extract_hospital_name(text):
    text = text or ''
    normalized_text = ' ' + normalize_match_text(text) + ' '
    for canonical_name, aliases in sorted(KNOWN_HOSPITAL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        for alias in sorted(aliases, key=len, reverse=True):
            normalized_alias = normalize_match_text(alias)
            if normalized_alias and f' {normalized_alias} ' in normalized_text:
                return canonical_name
    for pattern in HOSPITAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            value = clean_extracted_value(match.group(1))
            normalized_value = normalize_match_text(value)
            if value and len(value) <= 80 and 1 <= len(value.split()) <= 8 and normalized_value != normalize_match_text(text):
                return value
    for line in text.splitlines()[:8]:
        value = clean_extracted_value(line)
        lowered = value.lower()
        if value and len(value) <= 80 and 1 <= len(value.split()) <= 8 and any(token in lowered for token in ['hospital', 'clinic', 'medical center', 'medical centre', 'healthcare']):
            return value
    return 'Medical Center'
 
def normalize_medications(value): 
    medications = [] 
    if not isinstance(value, list): 
        return medications 
    for item in value: 
        if isinstance(item, str): 
            item = {'name': item} 
        if not isinstance(item, dict): 
            continue 
        medication = { 
            'name': str(item.get('name', '')).strip(), 
            'indication': str(item.get('indication', '')).strip(), 
            'dosage': str(item.get('dosage', '')).strip(), 
            'frequency': str(item.get('frequency', '')).strip(), 
            'duration': str(item.get('duration', '')).strip(), 
            'route': str(item.get('route', '')).strip(), 
            'instructions': str(item.get('instructions', '')).strip(), 
            'status': str(item.get('status', 'insufficient_information')).strip() or 'insufficient_information', 
        } 
        if any(medication.values()): 
            medications.append(medication) 
    return medications

def normalize_educational_examples(value):
    examples = []
    if not isinstance(value, list):
        return examples
    for item in value:
        if isinstance(item, str):
            item = {'name': item}
        if not isinstance(item, dict):
            continue
        example = {
            'name': str(item.get('name', '')).strip(),
            'use': str(item.get('use', '')).strip(),
            'warning': str(item.get('warning', '')).strip(),
        }
        if example['name'] and example['name'].lower() not in [entry['name'].lower() for entry in examples]:
            examples.append(example)
    return examples

def build_educational_tablet_examples(signals):
    examples = []
    for condition in signals.get('conditions', []):
        for item in EDUCATIONAL_TABLET_OPTIONS.get(condition, []):
            name = item.get('name', '').strip()
            if name and name.lower() not in [entry['name'].lower() for entry in examples]:
                examples.append(dict(item))
    return examples

def build_precautions(signals, medications=None):
    precautions = [
        'Confirm allergies, contraindications, drug interactions, renal and hepatic function, and pregnancy status where relevant before issuing the final prescription.',
        'Match every tablet, dose, frequency, and duration against the treating clinician\'s latest assessment and the most recent report values.',
    ]
    conditions = signals.get('conditions', []) if isinstance(signals, dict) else []
    tests = signals.get('tests', []) if isinstance(signals, dict) else []
    medications = medications or []

    if tests:
        precautions.append('Review the available ' + ', '.join(tests) + ' before finalizing the medication plan.')
    if 'kidney disease' in conditions:
        precautions.append('Check renal dosing and avoid nephrotoxic medicines unless the clinician has specifically approved them.')
    if 'liver disease' in conditions:
        precautions.append('Check hepatic dosing and avoid medicines with liver-related contraindications unless cleared by the clinician.')
    if 'diabetes' in conditions:
        precautions.append('Confirm the patient understands glucose monitoring, meal timing, and when to seek care for hypo- or hyperglycemia symptoms.')
    if 'hypertension' in conditions:
        precautions.append('Recheck blood pressure trends and symptoms such as dizziness before confirming antihypertensive therapy.')
    if any('food' in (item.get('instructions') or '').lower() for item in medications if isinstance(item, dict)):
        precautions.append('Counsel the patient to follow the meal-related instructions exactly as written for each tablet.')

    deduped = []
    for item in precautions:
        cleaned = clean_extracted_value(item)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped

def extract_report_signals(text):
    lowered = (text or '').lower()
    conditions = []
    for condition, keywords in CONDITION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            conditions.append(condition)
    tests = []
    for test_name, keywords in TEST_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tests.append(test_name)
    return {'conditions': conditions, 'tests': tests}

def extract_diagnosis_basis(text, signals):
    lines = [clean_extracted_value(line) for line in (text or '').splitlines()]
    basis = []
    for raw_line in lines:
        line = raw_line.lower()
        for header in DIAGNOSIS_HEADERS:
            if line.startswith(header):
                value = clean_extracted_value(raw_line.split(':', 1)[1] if ':' in raw_line else raw_line[len(header):])
                if value and value not in basis:
                    basis.append(value)
    for condition in signals.get('conditions', []):
        pretty = condition[0].upper() + condition[1:]
        if pretty not in basis:
            basis.append(pretty)
    return basis

def _looks_like_medication_line(line):
    lowered = line.lower()
    has_unit = any(re.search(r'\b\d+(?:\.\d+)?\s*' + unit + r'\b', lowered) for unit in DOSAGE_UNITS)
    has_frequency = any(hint in lowered for hint in FREQUENCY_HINTS)
    has_tablet_words = any(word in lowered for word in ['tab', 'tablet', 'cap', 'capsule', 'syrup', 'inj', 'injection'])
    return has_unit or has_frequency or has_tablet_words

def _extract_medication_name(line):
    cleaned = re.sub(r'^[\-\*\d\.\)\(]+\s*', '', line).strip()
    cleaned = re.sub(r'^(tab|tablet|cap|capsule|syrup|inj|injection)\s+', '', cleaned, flags=re.IGNORECASE)
    match = re.match(r'([A-Za-z][A-Za-z0-9\-/ ]{1,40}?)\s+(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units)\b|od\b|bd\b|tds\b|qid\b|hs\b)', cleaned, flags=re.IGNORECASE)
    if match:
        return clean_extracted_value(match.group(1))
    words = cleaned.split()
    if not words:
        return ''
    return clean_extracted_value(' '.join(words[:2]))

def _extract_frequency(line):
    lowered = line.lower()
    for hint in ['once daily', 'twice daily', 'thrice daily', 'every morning', 'every night', 'before food', 'after food', 'with food', 'od', 'bd', 'tds', 'qid', 'hs', 'stat']:
        if hint in lowered:
            return hint.upper() if hint in ['od', 'bd', 'tds', 'qid', 'hs', 'stat'] else hint
    return 'insufficient information'

def _extract_duration(line):
    match = re.search(r'(\bfor\s+\d+\s+(?:day|days|week|weeks|month|months)\b)', line, flags=re.IGNORECASE)
    if match:
        return clean_extracted_value(match.group(1))
    return 'insufficient information'

def _extract_dosage(line):
    match = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units))', line, flags=re.IGNORECASE)
    if match:
        return clean_extracted_value(match.group(1))
    return 'insufficient information'

def _extract_route(line):
    lowered = line.lower()
    if 'iv' in lowered or 'intravenous' in lowered:
        return 'intravenous'
    if 'im' in lowered or 'intramuscular' in lowered:
        return 'intramuscular'
    if 'oral' in lowered or any(word in lowered for word in ['tab', 'tablet', 'cap', 'capsule', 'syrup']):
        return 'oral'
    return 'insufficient information'

def _extract_instruction_summary(line):
    lowered = line.lower()
    instructions = []
    if 'before food' in lowered:
        instructions.append('Take before food')
    elif 'after food' in lowered:
        instructions.append('Take after food')
    elif 'with food' in lowered:
        instructions.append('Take with food')

    if 'every night' in lowered or 'hs' in lowered:
        instructions.append('preferably at night')
    elif 'every morning' in lowered:
        instructions.append('preferably in the morning')

    if 'stat' in lowered:
        instructions.append('start immediately')

    if not instructions:
        return 'Take exactly as directed by the clinician.'
    return '. '.join(instructions) + '.'

def extract_medications_from_text(text, signals):
    medications = []
    lines = [clean_extracted_value(line) for line in (text or '').splitlines() if clean_extracted_value(line)]
    in_med_section = False
    for line in lines:
        lowered = line.lower()
        if any(header == lowered.rstrip(':') for header in MEDICATION_SECTION_HEADERS):
            in_med_section = True
            continue
        if in_med_section and any(lowered.startswith(header) for header in DIAGNOSIS_HEADERS + ['advice', 'investigation', 'follow up', 'diet']):
            in_med_section = False
        if not in_med_section and not _looks_like_medication_line(line):
            continue
        if not _looks_like_medication_line(line):
            continue

        medication = {
            'name': _extract_medication_name(line),
            'indication': ', '.join(signals.get('conditions', [])[:2]) or 'insufficient information',
            'dosage': _extract_dosage(line),
            'frequency': _extract_frequency(line),
            'duration': _extract_duration(line),
            'route': _extract_route(line),
            'instructions': _extract_instruction_summary(line),
            'status': 'report-derived',
        }
        if medication['name'] and medication['name'].lower() not in [item['name'].lower() for item in medications]:
            medications.append(medication)
    return medications

def build_diet_recommendations(signals):
    recommendations = []
    for condition in signals.get('conditions', []):
        for advice in DIET_GUIDANCE_RULES.get(condition, []):
            if advice not in recommendations:
                recommendations.append(advice)
    if not recommendations:
        recommendations.append('Maintain regular meals, good hydration, and a balanced diet unless the treating clinician has advised restrictions.')
    return recommendations

def build_lifestyle_recommendations(signals):
    advice = []
    if 'diabetes' in signals.get('conditions', []):
        advice.append('Keep meal timing, physical activity, and glucose checks as consistent as possible.')
    if 'hypertension' in signals.get('conditions', []):
        advice.append('Check blood pressure regularly and limit tobacco and alcohol exposure if relevant.')
    if 'kidney disease' in signals.get('conditions', []):
        advice.append('Follow fluid and diet limits exactly as advised after kidney test review.')
    if not advice:
        advice.append('Follow the discharge advice, take medicines on time, and attend follow-up visits.')
    return advice

def build_test_based_monitoring(signals):
    monitoring = []
    for test_name in signals.get('tests', []):
        monitoring.append('Review ' + test_name + ' during follow-up and compare with earlier results if available.')
    if not monitoring:
        monitoring.append('Review symptoms, vitals, and any pending test results during follow-up.')
    return monitoring
 
def build_fallback_payload(patient_summary, patient_details=None, hospital_name='Medical Center'): 
    return { 
        'hospital_name': clean_extracted_value(hospital_name) or 'Medical Center',
        'patient_details': normalize_patient_details({}, fallback=patient_details),
        'patient_summary': shorten_patient_summary(patient_summary.strip() or 'Insufficient information in the report to summarize the patient.'), 
        'diagnosis_basis': [], 
        'tests_reviewed': [],
        'medications': [], 
        'educational_tablet_examples': [],
        'precautions': [
            'Confirm allergies, contraindications, drug interactions, renal and hepatic function, and pregnancy status where relevant before issuing the final prescription.',
            'Match every tablet, dose, frequency, and duration against the treating clinician\'s latest assessment and the most recent report values.',
        ],
        'non_medication_care': [ 
            'Insufficient information to generate a medication plan safely.', 
            'Use the uploaded report summary to support clinician review and manual prescribing.', 
        ], 
        'food_habits': ['Maintain regular meals and follow clinician-specific diet advice based on the report findings.'],
        'lifestyle_plan': ['Take medicines exactly as prescribed and attend follow-up as advised.'],
        'monitoring': [ 
            'Verify vitals, allergies, interaction risks, and recent labs before prescribing.', 
        ], 
        'warnings': list(DEFAULT_WARNINGS), 
        'missing_information': list(DEFAULT_MISSING_INFORMATION), 
        'review_required': 'A licensed clinician must confirm the indication, drug choice, dose, route, duration, interactions, and contraindications before prescribing.', 
    } 
 
def normalize_payload(payload, patient_summary, patient_details=None, hospital_name='Medical Center'): 
    data = payload if isinstance(payload, dict) else {} 
    fallback = build_fallback_payload(patient_summary, patient_details=patient_details, hospital_name=hospital_name) 
    normalized = { 
        'hospital_name': clean_extracted_value(data.get('hospital_name', '')) or fallback['hospital_name'],
        'patient_details': normalize_patient_details(data.get('patient_details'), fallback=fallback['patient_details']),
        'patient_summary': shorten_patient_summary(str(data.get('patient_summary', '')).strip() or fallback['patient_summary']), 
        'diagnosis_basis': normalize_list(data.get('diagnosis_basis')), 
        'tests_reviewed': normalize_list(data.get('tests_reviewed')),
        'medications': normalize_medications(data.get('medications')), 
        'educational_tablet_examples': normalize_educational_examples(data.get('educational_tablet_examples')),
        'precautions': normalize_list(data.get('precautions')) or fallback['precautions'],
        'non_medication_care': normalize_list(data.get('non_medication_care')), 
        'food_habits': normalize_list(data.get('food_habits')) or fallback['food_habits'],
        'lifestyle_plan': normalize_list(data.get('lifestyle_plan')) or fallback['lifestyle_plan'],
        'monitoring': normalize_list(data.get('monitoring')), 
        'warnings': normalize_list(data.get('warnings')) or fallback['warnings'], 
        'missing_information': normalize_list(data.get('missing_information')) or fallback['missing_information'], 
        'review_required': str(data.get('review_required', '')).strip() or fallback['review_required'], 
    } 
    for warning in DEFAULT_WARNINGS: 
        if warning not in normalized['warnings']: 
            normalized['warnings'].append(warning) 
    return normalized
 
def build_prescription_text(payload): 
    lines = [] 
    lines.append(payload.get('hospital_name') or 'Medical Center')
    lines.append('')
    lines.append('Patient details:')
    patient_details = payload.get('patient_details') or {}
    for field in PATIENT_DETAIL_FIELDS:
        lines.append(PATIENT_DETAIL_LABELS[field] + ': ' + (patient_details.get(field) or 'Not provided'))
    lines.append('')
    lines.append('Patient snapshot:') 
    patient_snapshot = clean_extracted_value(payload.get('patient_summary') or 'Insufficient information')
    snapshot_sentences = re.split(r'(?<=[.!?])\s+', patient_snapshot)
    lines.append(' '.join(snapshot_sentences[:2]).strip() or patient_snapshot) 
    lines.append('') 
    lines.append('Draft prescription for clinician review:') 
    medications = payload.get('medications') or [] 
    if medications: 
        for index, medication in enumerate(medications, start=1): 
            lines.append(str(index) + '. ' + (medication.get('name') or 'Medication not specified')) 
            lines.append('   Indication: ' + (medication.get('indication') or 'insufficient information')) 
            lines.append('   Dosage: ' + (medication.get('dosage') or 'insufficient information')) 
            lines.append('   Frequency: ' + (medication.get('frequency') or 'insufficient information')) 
            lines.append('   Duration: ' + (medication.get('duration') or 'insufficient information')) 
            lines.append('   Route: ' + (medication.get('route') or 'insufficient information')) 
            lines.append('   Instructions: ' + (medication.get('instructions') or 'insufficient information')) 
            lines.append('   Status: ' + (medication.get('status') or 'insufficient_information')) 
    else: 
        lines.append('No medication recommendation generated because the report did not include enough prescribing details.') 
    educational_examples = payload.get('educational_tablet_examples') or []
    lines.append('')
    lines.append('General educational tablet examples (not a prescription):')
    if educational_examples:
        for index, example in enumerate(educational_examples, start=1):
            lines.append(str(index) + '. ' + (example.get('name') or 'Example not specified'))
            lines.append('   General use: ' + (example.get('use') or 'general educational example'))
            lines.append('   Warning: ' + (example.get('warning') or 'Educational example only. A clinician must decide whether it is appropriate.'))
    else:
        lines.append('No general educational tablet examples were added because the report did not clearly match a supported condition.')
    for section_title, items in [('Diagnosis:', payload.get('diagnosis_basis')), ('Tests reviewed:', payload.get('tests_reviewed')), ('Precautions:', payload.get('precautions')), ('Food habits:', payload.get('food_habits')), ('Lifestyle plan:', payload.get('lifestyle_plan')), ('Non-medication care:', payload.get('non_medication_care')), ('Monitoring:', payload.get('monitoring')), ('Missing information before prescribing:', payload.get('missing_information')), ('Safety warnings:', payload.get('warnings'))]: 
        lines.append('') 
        lines.append(section_title) 
        if items: 
            for item in items: 
                lines.append('- ' + item) 
        else: 
            lines.append('- None provided') 
    lines.append('') 
    lines.append('Review required:') 
    lines.append(payload.get('review_required') or 'Clinician review required.') 
    return '\n'.join(lines)

def build_prescription_pdf(payload):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError('PDF export requires reportlab to be installed.') from exc

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left_margin = 40
    top_margin = 50
    line_height = 14
    max_chars = 100
    y_position = height - top_margin

    text_lines = build_prescription_text(payload).splitlines()
    for raw_line in text_lines:
        wrapped_lines = textwrap.wrap(raw_line, width=max_chars) or ['']
        for line in wrapped_lines:
            if y_position <= top_margin:
                pdf.showPage()
                y_position = height - top_margin
            pdf.drawString(left_margin, y_position, line)
            y_position -= line_height

    pdf.save()
    return buffer.getvalue()
 
def generate_prescription_result(summarizer, text, on_premise=False, vector_store=None): 
    try:
        text_clean = summarizer._preprocess(text) 
        seed = summarizer.textrank_extract(text_clean, top_k=6) 
        patient_summary = seed or summarizer.textrank_extract(text_clean, top_k=3) or text_clean[:500] 
        patient_details = extract_patient_details(text_clean)
        hospital_name = extract_hospital_name(text_clean)
        signals = extract_report_signals(text_clean)
        retrieved = [] 
        if vector_store is not None: 
            try: 
                available_chunks = len(getattr(vector_store, 'texts', []) or [])
                top_k = max(2, min(4, available_chunks or 4))
                retrieved = vector_store.search(seed or text_clean, top_k=top_k) 
            except Exception as exc: 
                logger.warning('Vector retrieval failed for prescription draft: %s', exc) 
                retrieved = [] 
        if not retrieved: 
            chunks = summarizer.chunk_text(text_clean, max_words=400, overlap_words=50) 
            retrieved = [(chunk, {'chunk_id': index, 'source': 'fallback'}) for index, chunk in enumerate(chunks[:4])] 
        sources = [] 
        for chunk_text, metadata in retrieved: 
            points = summarizer.chunk_to_bullets(chunk_text, top_k=4) 
            if len(chunk_text) > 400: 
                snippet = chunk_text[:400] + '...' 
            else: 
                snippet = chunk_text 
            sources.append({'snippet': snippet, 'points': points, 'meta': metadata}) 
        payload = build_fallback_payload(patient_summary, patient_details=patient_details, hospital_name=hospital_name) 
        payload['diagnosis_basis'] = extract_diagnosis_basis(text_clean, signals)
        payload['tests_reviewed'] = list(signals.get('tests', []))
        payload['medications'] = extract_medications_from_text(text_clean, signals)
        payload['educational_tablet_examples'] = build_educational_tablet_examples(signals)
        payload['precautions'] = build_precautions(signals, payload['medications'])
        payload['food_habits'] = build_diet_recommendations(signals)
        payload['lifestyle_plan'] = build_lifestyle_recommendations(signals)
        payload['monitoring'] = build_test_based_monitoring(signals)
        model_name = 'embedding-guided-prescription-fallback' if getattr(summarizer, 'embedder', None) else 'prescription-fallback' 
        if summarizer.openai and not on_premise and retrieved: 
            context_parts = [] 
            for index, item in enumerate(retrieved, start=1): 
                chunk_text = item[0] 
                metadata = item[1] 
                chunk_id = metadata.get('chunk_id', index - 1) 
                context_parts.append('[chunk ' + str(chunk_id) + '] ' + chunk_text) 
            prompt = (
                'You are assisting a licensed clinician. Use only the provided context from the uploaded report. '
                'Do not invent hospital name, patient demographics, diagnoses, medications, doses, durations, routes, or contraindications. '
                'If a field is not present in the report, return "Not provided" for patient details and leave medication fields as "insufficient information". '
                'Return JSON only with keys hospital_name, patient_details, patient_summary, diagnosis_basis, tests_reviewed, medications, precautions, food_habits, lifestyle_plan, non_medication_care, monitoring, warnings, missing_information, review_required. '
                'patient_details must be an object with keys name, age, sex, dob, patient_id, doctor. '
                'For medications, include how the patient should take them if the report provides timing or meal-related instructions. '
                'For precautions, provide short clinician-review safety checks tied to the report and proposed tablets. '
                'For food_habits and lifestyle_plan, base the advice only on diseases, symptoms, and tests explicitly mentioned in the report.'
                + '\n\nContext:\n' + '\n\n'.join(context_parts)
            )
            try: 
                raw_output = summarizer.abstractive_openai(prompt, max_tokens=900) 
                payload = normalize_payload(
                    json.loads(extract_json_object(raw_output)),
                    patient_summary,
                    patient_details=patient_details,
                    hospital_name=hospital_name,
                ) 
                payload['educational_tablet_examples'] = build_educational_tablet_examples(signals)
                model_name = 'openai-prescription' 
            except Exception as exc: 
                logger.warning('Prescription draft generation failed, using fallback: %s', exc) 
        text_output = build_prescription_text(payload) 
        return {'summary': text_output, 'seed': seed, 'sources': sources, 'model': model_name, 'prescription': payload}
    except Exception as exc:
        logger.exception('Prescription draft generation failed completely')
        fallback_details = extract_patient_details(text)
        fallback_hospital = extract_hospital_name(text)
        fallback_payload = build_fallback_payload(
            (text or '').strip()[:500],
            patient_details=fallback_details,
            hospital_name=fallback_hospital,
        )
        fallback_signals = extract_report_signals(text)
        fallback_payload['diagnosis_basis'] = extract_diagnosis_basis(text, fallback_signals)
        fallback_payload['tests_reviewed'] = list(fallback_signals.get('tests', []))
        fallback_payload['medications'] = extract_medications_from_text(text, fallback_signals)
        fallback_payload['educational_tablet_examples'] = build_educational_tablet_examples(fallback_signals)
        fallback_payload['precautions'] = build_precautions(fallback_signals, fallback_payload['medications'])
        fallback_payload['food_habits'] = build_diet_recommendations(fallback_signals)
        fallback_payload['lifestyle_plan'] = build_lifestyle_recommendations(fallback_signals)
        fallback_payload['monitoring'] = build_test_based_monitoring(fallback_signals)
        return {
            'summary': build_prescription_text(fallback_payload),
            'seed': '',
            'sources': [],
            'model': 'prescription-fallback',
            'prescription': fallback_payload,
            'error': str(exc),
        }
