from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat
from typing import Optional, Dict
from math import pow
import uvicorn

# -------------------------
# Configuration & IS tables
# -------------------------
WATER_CONTENT_50MM = {10: 208.0, 12.5: 195.0, 20: 186.0, 40: 165.0}
AIR_CONTENT_PCT = {10: 1.5, 12.5: 1.2, 20: 1.0, 40: 0.8}
SLUMP_CORRECTION_PERCENT_PER_25MM = 3.0
DEFAULT_SP_REDUCTION_PCT = 20.0

# Recommended Slump Ranges (mm) based on IS 10262 Table 4 (simplified)
PLACING_SLUMP_RANGES = {
    "manual_placing": (50, 100),         # Normal reinforcement, shallow sections
    "chute_non_pumpable": (75, 125),     # Heavily reinforced/vibrated sections
    "pumping": (100, 150),               # Pumping requires higher flow
}

EMISSION_FACTORS = {
    "cement": 0.90,
    "flyash": 0.05,
    "ggbs": 0.07,
    "silicafume": 0.12,
    "metakaolin": 0.10,
    "rha": 0.05,
    "coconut_ash": 0.05,
    "sugarcane_ash": 0.05,
    "sand": 0.005,
    "coarse": 0.01,
    "water": 0.002
}

IS456_DURABILITY_LIMITS_REINFORCED = {
    "mild":      {"min_cement": 300, "max_w_c": 0.50},
    "moderate": {"min_cement": 300, "max_w_c": 0.50},
    "severe":    {"min_cement": 320, "max_w_c": 0.45},
    "very_severe":{"min_cement": 340, "max_w_c": 0.45},
    "extreme":   {"min_cement": 360, "max_w_c": 0.40}
}

# Strength Factor based on Cement Grade/Type (Simplified model)
STRENGTH_FACTOR = {
    ("OPC", 43): 1.00,
    ("OPC", 53): 1.15,
    ("PPC-FLYASH", 0): 0.90,
    ("PPC-CALCINED", 0): 0.95,
    ("PSC", 0): 0.85,
}

# Standard Deviation (S) for various grades, based on IS 10262 Table 2
def get_standard_deviation(fck_mpa: float) -> float:
    if fck_mpa >= 50:
        return 6.0
    if fck_mpa >= 30:
        return 5.0
    if fck_mpa >= 25:
        return 4.0
    return 3.5 # For M20 and below

# -------------------------
# Helper functions
# -------------------------
def classify_workability(slump_mm: float) -> str:
    if slump_mm < 50:
        return "Low"
    if slump_mm <= 100:
        return "Medium"
    if slump_mm <= 150:
        return "High"
    return "Very High"

def target_mean_strength_is10262(fck_mpa: float) -> float:
    """
    Calculates the target average compressive strength (f'ck) based on IS 10262:2019, C-3.
    """
    S = get_standard_deviation(fck_mpa)
    X = 6.5 
    
    val_a = fck_mpa + 1.65 * S
    val_b = fck_mpa + X
    
    return max(val_a, val_b)

def predict_strength_adjusted(fck_target: float, cement_type: str, cement_grade: int) -> float:
    """
    Predicts a realistic 28-day strength based on the IS 10262 Target Mean Strength (f'ck), 
    adjusted only by cement type/grade factor.
    """
    
    fcm_is_target = target_mean_strength_is10262(fck_target)

    predicted_strength = fcm_is_target
    
    key = (cement_type.upper(), cement_grade)
    cement_factor = STRENGTH_FACTOR.get(key, STRENGTH_FACTOR.get((cement_type.upper(), 0), 1.0))
    predicted_strength *= cement_factor
    
    predicted_strength = max(fck_target, min(predicted_strength, 100.0))

    return predicted_strength

def compute_mix_from_inputs(
    fck_mpa: float,
    w_c_ratio: float,
    slump_mm: float,
    max_agg_size_mm: int,
    replacements: Dict[str, float],
    use_superplasticizer: bool,
    exposure: str,
    cement_type: str,
    cement_grade: int,
    placing_method: str,
    
    # NEW MATERIAL INPUTS
    sg_cement: float,
    sg_coarse: float,
    sg_fine: float,
    wa_coarse: float,
    wa_fine: float,
    scm_sg: Optional[Dict[str, float]] = None, # Dynamic SCM Specific Gravity
    
    # PROJECT INPUTS
    project_volume_m3: float = 1.0,
    wastage_pct: float = 0.0
):
    """
    Core IS10262-driven (simplified) calculation for one mix (per m3).
    Also calculates total required material mass for project volume + wastage.
    """
    
    # 0) Consolidate Specific Gravities for calculation
    SG = {
        "cement": sg_cement,
        "coarse": sg_coarse,
        "fine": sg_fine,
        "water": 1.0
    }
    # Add dynamic SCM specific gravities (use a default of 2.7 if not provided for an active SCM)
    SCM_SG = {k: v if v else 2.70 for k, v in (scm_sg or {}).items()}
    # Use 2.7 as the default for volume calculation for any remaining SCMs
    default_scm_sg = 2.70

    # Validate water table availability
    if max_agg_size_mm not in WATER_CONTENT_50MM:
        raise ValueError(f"Max aggregate size {max_agg_size_mm} not supported. Supported: {list(WATER_CONTENT_50MM.keys())}")
    
    # 0.1) Slump Placability Check (for warning, not changing the calculation)
    min_slump, max_slump = PLACING_SLUMP_RANGES.get(placing_method, (50, 150))
    placability_warning = None
    if slump_mm < min_slump:
        placability_warning = f"Slump ({slump_mm} mm) is too Low for {placing_method.replace('_', ' ').title()}. IS 10262 suggests min {min_slump} mm."
    elif slump_mm > max_slump:
        placability_warning = f"Slump ({slump_mm} mm) is too High for {placing_method.replace('_', ' ').title()}. This may cause segregation/bleeding. IS 10262 suggests max {max_slump} mm."

    # 1) preliminary w/c
    w_c = w_c_ratio

    # 2) water content for 50 mm slump
    water_50mm = WATER_CONTENT_50MM[max_agg_size_mm]

    # 3) slump adjustment
    delta_slump = slump_mm - 50.0
    correction_pct = (delta_slump / 25.0) * SLUMP_CORRECTION_PERCENT_PER_25MM
    water_required_mass = water_50mm * (1.0 + correction_pct / 100.0)

    # 4) SP reduction
    if use_superplasticizer:
        water_required_mass *= (1.0 - DEFAULT_SP_REDUCTION_PCT / 100.0)

    # 5) cementitious total
    cementitious_total_mass = water_required_mass / w_c

    # 6) split replacements
    scm_masses = {}
    total_scm_fraction = 0.0
    if replacements:
        for k, frac in replacements.items():
            scm_masses[k] = cementitious_total_mass * frac
            total_scm_fraction += frac
            
    if total_scm_fraction >= 1.0:
        raise ValueError("Total replacement fraction cannot be 1.0 or greater.")
            
    cement_actual_mass = cementitious_total_mass * (1.0 - total_scm_fraction)

    # 7) IS-456 enforcement (min cement & max w/c)
    limits = IS456_DURABILITY_LIMITS_REINFORCED.get(exposure.lower(), IS456_DURABILITY_LIMITS_REINFORCED["moderate"])
    min_cement_req = limits["min_cement"]
    max_w_c_allowed = limits["max_w_c"]
    compliance_w_c = (w_c <= max_w_c_allowed)
    compliance_cement = (cement_actual_mass >= min_cement_req)
    warnings = []
    
    if placability_warning:
        warnings.append(placability_warning)
        
    if not compliance_w_c:
        warnings.append(f"Selected w/c = {w_c:.3f} exceeds IS-456 max {max_w_c_allowed:.3f} for {exposure.title()} exposure.")
    
    if not compliance_cement:
        warnings.append(f"Cement {cement_actual_mass:.1f} kg/m³ < IS-456 minimum {min_cement_req} kg/m³ for {exposure.title()}. Raising cement content.")
        
        # Adjust to minimum cement
        cement_actual_mass = float(min_cement_req)
        
        if (1.0 - total_scm_fraction) <= 0:
             raise ValueError("Total SCM fraction too high (>=1.0)")
             
        cementitious_total_mass = cement_actual_mass / (1.0 - total_scm_fraction)
        
        # Recalculate SCM masses based on the new, higher cementitious total
        scm_masses = {}
        for k, frac in replacements.items():
             scm_masses[k] = cementitious_total_mass * frac
        
        # Recalculate water based on the original w/c ratio and the new, higher cementitious total
        water_required_mass = cementitious_total_mass * w_c

    # 8) air fraction
    air_pct = AIR_CONTENT_PCT.get(max_agg_size_mm, 1.0)
    vol_air = air_pct / 100.0

    # 9) volumes (using user-defined SG)
    unit_mass = 1000.0 # kg/m3 (for density of water)
    
    vol_water = water_required_mass / unit_mass
    vol_cement = cement_actual_mass / (unit_mass * SG["cement"])
    
    vol_scms = 0.0
    for scmat, mass in scm_masses.items():
        sg_scm = SCM_SG.get(scmat.lower(), default_scm_sg) # Use user SG or default
        vol_scms += mass / (unit_mass * sg_scm)
        
    vol_binders = vol_cement + vol_scms

    vol_aggregates = 1.0 - (vol_water + vol_binders + vol_air)
    if vol_aggregates <= 0:
        raise ValueError(f"Negative aggregate volume ({vol_aggregates:.4f}) - check inputs.")

    # 10) aggregate distribution (default fine=40% of aggregate volume)
    fine_vol = 0.40 * vol_aggregates
    coarse_vol = vol_aggregates - fine_vol
    
    # Mass of aggregates (Saturated Surface Dry - SSD basis)
    mass_fine_ssd = fine_vol * unit_mass * SG["fine"]
    mass_coarse_ssd = coarse_vol * unit_mass * SG["coarse"]
    
    # 10.1) Water absorption correction (IS 10262)
    # Water absorbed by aggregates from mix (mass_ssd * WA%)
    water_absorbed_coarse = mass_coarse_ssd * (wa_coarse / 100.0)
    water_absorbed_fine = mass_fine_ssd * (wa_fine / 100.0)
    
    # Total water correction: Subtract the water absorbed by the aggregates from the free water mass.
    final_free_water = water_required_mass - (water_absorbed_coarse + water_absorbed_fine)

    if final_free_water < 0:
        warnings.append(f"Aggregate Water Absorption is high ({wa_coarse}% C.A, {wa_fine}% F.A). Free water requirement is negative. Check inputs.")
        final_free_water = 0.0

    # Final masses for reporting (Aggregates are batched as SSD for volume calc, so mass is SSD mass)
    mass_fine = mass_fine_ssd
    mass_coarse = mass_coarse_ssd
    
    # 11) predicted strength (IS 10262 based)
    predicted_fc_28 = predict_strength_adjusted(fck_mpa, cement_type, cement_grade)

    # 12) co2 estimates (use final_free_water for eco mix)
    ef = EMISSION_FACTORS
    
    # BASELINE: Pure OPC mix calculation (using the cementitious total mass)
    baseline_cement_total = cementitious_total_mass 
    co2_base = baseline_cement_total * ef["cement"] + final_free_water * ef["water"] + mass_fine * ef["sand"] + mass_coarse * ef["coarse"]
    
    # ECO MIX: Calculation using actual cement and SCMs
    co2_eco = cement_actual_mass * ef["cement"] + final_free_water * ef["water"] + mass_fine * ef["sand"] + mass_coarse * ef["coarse"]
    for scmat, mass in scm_masses.items():
        key = scmat.lower()
        map_keys = {
            "rha":"rha", "coconut_ash":"coconut_ash", "sugarcane_ash":"sugarcane_ash", 
            "ggbs":"ggbs", "silicafume":"silicafume", "metakaolin":"metakaolin", "flyash":"flyash"
        }
        ef_key = map_keys.get(key, key)
        sc_ef = ef.get(ef_key, 0.05)
        co2_eco += mass * sc_ef

    co2_reduction_pct = ((co2_base - co2_eco) / co2_base * 100.0) if co2_base > 0 else 0.0

    # 13) workability
    work_class = classify_workability(slump_mm)

    # 14) durability index (heuristic)
    durability_index = max(20, min(100, (1.0 - w_c) * 100))
    durability_index += (total_scm_fraction * 10)
    exposure_penalty = {"mild": 0, "moderate": -5, "severe": -12, "very_severe": -18, "extreme": -25}
    durability_index += exposure_penalty.get(exposure.lower(), -5)
    durability_index = max(20, min(100, durability_index))


    # 15) Calculate Project Quantities
    project_factor = project_volume_m3 * (1.0 + wastage_pct / 100.0)
    
    project_quantities = {
        "cement_actual": round(cement_actual_mass * project_factor, 2),
        "water": round(final_free_water * project_factor, 2),
        "fine_aggregate": round(mass_fine * project_factor, 2),
        "coarse_aggregate": round(mass_coarse * project_factor, 2),
        
        # SCMs (include 0 values for consistency)
        **{k: round(v * project_factor, 2) for k, v in scm_masses.items()},
    }


    # 16) assemble result
    result = {
        "mix_masses_kg_per_m3": {
            "cement_actual": round(cement_actual_mass, 2),
            **{k: round(v, 2) for k, v in scm_masses.items()},
            "water": round(final_free_water, 2),
            "fine_aggregate": round(mass_fine, 2),
            "coarse_aggregate": round(mass_coarse, 2),
            "air_pct": round(air_pct, 2)
        },
        "predictions": {
            "predicted_28d_strength_MPa": round(predicted_fc_28, 2),
            "workability_class": work_class,
            "durability_index_0_100": round(durability_index, 1)
        },
        "co2": {
            "co2_base_kg_per_m3": round(co2_base, 2),
            "co2_eco_kg_per_m3": round(co2_eco, 2),
            "co2_reduction_pct": round(co2_reduction_pct, 2)
        },
        "is_compliance": {
            "w_c_within_IS456": compliance_w_c,
            "cement_meets_minimum": compliance_cement,
            "warnings": warnings,
            "placability_warning": placability_warning
        },
        "required_project_quantities": project_quantities
    }

    return result

# -------------------------
# FastAPI + Models
# -------------------------
class StandardInput(BaseModel):
    project_volume_m3: confloat(gt=0) = 1.0
    wastage_pct: confloat(ge=0) = 0.0
    fck_mpa: confloat(gt=0)
    cement_type: str = Field("OPC")
    cement_grade: int = Field(43)
    w_c_ratio: confloat(gt=0)
    slump_mm: confloat(ge=0)
    exposure: str = Field("moderate")
    max_agg_size_mm: int = Field(20)
    use_superplasticizer: Optional[bool] = False
    placing_method: str = Field("manual_placing")
    
    # NEW MATERIAL INPUTS
    sg_cement: confloat(gt=0)
    sg_coarse: confloat(gt=0)
    sg_fine: confloat(gt=0)
    wa_coarse: confloat(ge=0)
    wa_fine: confloat(ge=0)

class EcoInput(BaseModel):
    project_volume_m3: confloat(gt=0) = 1.0
    wastage_pct: confloat(ge=0) = 0.0
    fck_mpa: confloat(gt=0)
    cement_type: str = Field("OPC")
    cement_grade: int = Field(43)
    w_c_ratio: confloat(gt=0)
    slump_mm: confloat(ge=0)
    exposure: str = Field("moderate")
    max_agg_size_mm: int = Field(20)
    use_superplasticizer: Optional[bool] = False
    placing_method: str = Field("manual_placing")

    # NEW MATERIAL INPUTS
    sg_cement: confloat(gt=0)
    sg_coarse: confloat(gt=0)
    sg_fine: confloat(gt=0)
    wa_coarse: confloat(ge=0)
    wa_fine: confloat(ge=0)
    scm_sg: Optional[Dict[str, confloat(gt=0)]] = Field(None)

    # SCM Replacements (fractions of cementitious mass)
    ggbs_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    flyash_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    silica_fume_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    rice_husk_ash_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    metakaolin_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    sugarcane_ash_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    

class CompareInput(BaseModel):
    standard: StandardInput
    eco: EcoInput

app = FastAPI(title="EcoMix Design API", version="1.0")

# Allow CORS for frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.post("/api/design/standard")
def design_standard(payload: StandardInput):
    try:
        replacements = {}
        # Standard mix uses the specified cement and grade, no SCMs
        res = compute_mix_from_inputs(
            fck_mpa=payload.fck_mpa,
            w_c_ratio=float(payload.w_c_ratio),
            slump_mm=float(payload.slump_mm),
            max_agg_size_mm=payload.max_agg_size_mm,
            replacements=replacements,
            use_superplasticizer=payload.use_superplasticizer,
            exposure=payload.exposure,
            cement_type=payload.cement_type,
            cement_grade=payload.cement_grade,
            placing_method=payload.placing_method,
            sg_cement=float(payload.sg_cement),
            sg_coarse=float(payload.sg_coarse),
            sg_fine=float(payload.sg_fine),
            wa_coarse=float(payload.wa_coarse),
            wa_fine=float(payload.wa_fine),
            scm_sg={}, # No SCMs in standard mix
            project_volume_m3=float(payload.project_volume_m3),
            wastage_pct=float(payload.wastage_pct),
        )
        res["inputs"] = payload.dict()
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/design/eco")
def design_eco(payload: EcoInput):
    try:
        replacements = {
            "ggbs": float(payload.ggbs_frac or 0.0),
            "flyash": float(payload.flyash_frac or 0.0),
            "silicafume": float(payload.silica_fume_frac or 0.0),
            "rha": float(payload.rice_husk_ash_frac or 0.0),
            "metakaolin": float(payload.metakaolin_frac or 0.0),
            "sugarcane_ash": float(payload.sugarcane_ash_frac or 0.0),
        }
        # remove zero entries
        replacements = {k: v for k, v in replacements.items() if v and v > 0.0}
        
        if sum(replacements.values()) >= 0.9:
            raise HTTPException(status_code=400, detail="Total replacement fraction must be less than 0.9")
            
        res = compute_mix_from_inputs(
            fck_mpa=payload.fck_mpa,
            w_c_ratio=float(payload.w_c_ratio),
            slump_mm=float(payload.slump_mm),
            max_agg_size_mm=payload.max_agg_size_mm,
            replacements=replacements,
            use_superplasticizer=payload.use_superplasticizer,
            exposure=payload.exposure,
            cement_type=payload.cement_type,
            cement_grade=payload.cement_grade,
            placing_method=payload.placing_method,
            sg_cement=float(payload.sg_cement),
            sg_coarse=float(payload.sg_coarse),
            sg_fine=float(payload.sg_fine),
            wa_coarse=float(payload.wa_coarse),
            wa_fine=float(payload.wa_fine),
            scm_sg={k: float(v) for k, v in (payload.scm_sg or {}).items()},
            project_volume_m3=float(payload.project_volume_m3),
            wastage_pct=float(payload.wastage_pct),
        )
        res["inputs"] = payload.dict()
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/design/compare")
def design_compare(payload: CompareInput):
    try:
        st = payload.standard
        eco = payload.eco
        
        # 1. Prepare SCM replacements for Eco Mix
        eco_replacements = {
            "ggbs": float(eco.ggbs_frac or 0.0),
            "flyash": float(eco.flyash_frac or 0.0),
            "silicafume": float(eco.silica_fume_frac or 0.0),
            "rha": float(eco.rice_husk_ash_frac or 0.0),
            "metakaolin": float(eco.metakaolin_frac or 0.0),
            "sugarcane_ash": float(eco.sugarcane_ash_frac or 0.0),
        }
        eco_replacements = {k: v for k, v in eco_replacements.items() if v and v > 0.0}
        
        scm_sg_data = {k: float(v) for k, v in (eco.scm_sg or {}).items()}

        # 1.1 Calculate Standard Mix (No SCM replacements)
        res_standard = compute_mix_from_inputs(
            fck_mpa=st.fck_mpa,
            w_c_ratio=float(st.w_c_ratio),
            slump_mm=float(st.slump_mm),
            max_agg_size_mm=st.max_agg_size_mm,
            replacements={}, 
            use_superplasticizer=st.use_superplasticizer,
            exposure=st.exposure,
            cement_type=st.cement_type,
            cement_grade=st.cement_grade,
            placing_method=st.placing_method,
            sg_cement=float(st.sg_cement),
            sg_coarse=float(st.sg_coarse),
            sg_fine=float(st.sg_fine),
            wa_coarse=float(st.wa_coarse),
            wa_fine=float(st.wa_fine),
            scm_sg={},
            project_volume_m3=float(st.project_volume_m3),
            wastage_pct=float(st.wastage_pct),
        )

        # 1.2 Calculate Eco Mix
        res_eco = compute_mix_from_inputs(
            fck_mpa=eco.fck_mpa,
            w_c_ratio=float(eco.w_c_ratio),
            slump_mm=float(eco.slump_mm),
            max_agg_size_mm=eco.max_agg_size_mm,
            replacements=eco_replacements,
            use_superplasticizer=eco.use_superplasticizer,
            exposure=eco.exposure,
            cement_type=eco.cement_type,
            cement_grade=eco.cement_grade,
            placing_method=eco.placing_method,
            sg_cement=float(eco.sg_cement),
            sg_coarse=float(eco.sg_coarse),
            sg_fine=float(eco.sg_fine),
            wa_coarse=float(eco.wa_coarse),
            wa_fine=float(eco.wa_fine),
            scm_sg=scm_sg_data,
            project_volume_m3=float(eco.project_volume_m3),
            wastage_pct=float(eco.wastage_pct),
        )

        # 3. Compute Comparisons
        co2_base = res_standard["co2"]["co2_base_kg_per_m3"]
        co2_eco = res_eco["co2"]["co2_eco_kg_per_m3"]
        co2_reduction_pct = round(((co2_base - co2_eco) / co2_base * 100.0) if co2_base > 0 else 0.0, 2)
        
        # Cement Saving % is based on the actual OPC component mass
        cement_std = res_standard["mix_masses_kg_per_m3"]["cement_actual"]
        cement_eco = res_eco["mix_masses_kg_per_m3"]["cement_actual"]
        cement_saving_pct = round(((cement_std - cement_eco) / cement_std * 100.0) if cement_std > 0 else 0.0, 2)

        return {
            "standard": res_standard,
            "eco": res_eco,
            "comparison": {
                "co2_reduction_pct": co2_reduction_pct,
                "cement_saving_pct": cement_saving_pct,
                "predicted_strength_diff_MPa": round(res_eco["predictions"]["predicted_28d_strength_MPa"] - res_standard["predictions"]["predicted_28d_strength_MPa"], 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check
@app.get("/api/health")
def health():
    return {"status":"ok"}

# Run server with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)