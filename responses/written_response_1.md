# Written Response 1 – Problem Definition

## SFHA Advanced Data + AI Program – Week 8 Capstone

---

### What is the problem?

Hospital readmissions—when a patient discharged from a hospital returns within
30 days—are one of the most costly and preventable events in modern healthcare.
In Canada and the United States, approximately **15–20% of discharged patients**
are readmitted within 30 days, costing the healthcare system billions of dollars
annually and placing significant strain on hospital resources. Beyond the
financial burden, readmissions signal gaps in care: a patient who returns so
quickly either was not ready to be discharged, did not receive adequate
follow-up support, or experienced a complication that better discharge planning
could have prevented.

This project addresses the following problem:

> **Can we predict, at the time of discharge, which patients are at high risk
> of returning to hospital within 30 days—so that care teams can intervene
> before it happens?**

---

### Why does this problem matter?

The consequences of unplanned readmissions are serious and multidimensional:

- **Patient outcomes**: Readmission often indicates deterioration. Patients who
  return to hospital are at elevated risk of complications, longer subsequent
  stays, and—in vulnerable populations—death.
- **Healthcare costs**: The Canadian Institute for Health Information (CIHI)
  estimates that preventable readmissions cost the Canadian health system over
  $1.8 billion annually. In the US, Medicare alone pays more than $26 billion
  per year on readmissions.
- **System capacity**: Every unnecessary readmission occupies a bed that could
  serve a new patient, contributing to hallway medicine and emergency department
  overcrowding.
- **Quality metrics**: Readmission rates are a primary quality indicator used
  by provincial and federal health authorities to evaluate hospital performance
  and to guide funding decisions.

---

### Who benefits?

A well-functioning readmission prediction model benefits multiple stakeholders:

1. **Patients and families**: High-risk patients can be identified early and
   given structured follow-up support (phone check-ins, home visits, specialist
   referrals) that reduces the likelihood of a preventable return.
2. **Care coordinators and nurses**: A risk score at discharge gives clinicians
   an actionable, data-driven priority list rather than relying solely on
   clinical intuition.
3. **Hospital administrators**: Reducing readmission rates improves quality
   scores, avoids financial penalties under pay-for-performance programmes, and
   frees up bed capacity.
4. **Payers and governments**: Preventable readmissions are expensive; any
   reduction translates directly into health system savings that can be
   reallocated to other services.

---

### What is a useful outcome?

A useful outcome is a **patient risk score** (0–100%) generated at the point
of discharge. Patients above a defined threshold (e.g., predicted probability
≥ 50%) are flagged for intensive discharge planning: medication reconciliation,
a scheduled 48-hour follow-up call, and a 7-day post-discharge telehealth
appointment. Over time, the model should be validated on real Electronic Health
Record (EHR) data and integrated into existing clinical workflows (e.g., as an
alert in the hospital's patient management system).

The model is not intended to replace clinical judgment; rather, it augments it
by surfacing patterns in large volumes of data that a busy clinician may not
have time to process manually. Success is measured not by model accuracy alone,
but by a **reduction in 30-day readmission rates** at institutions that adopt
the intervention.
