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

ASSUMED_CONCRETE_DENSITY = 2400.0
SG = {"cement": 3.15, "scm": 2.7, "fine": 2.65, "coarse": 2.70, "water": 1.0}

# Strength Factor based on Cement Grade/Type (Simplified model, 43 OPC is base=1.0)
STRENGTH_FACTOR = {
    ("OPC", 43): 1.0,
    ("OPC", 53): 1.15,
    ("PPC-FLYASH", 0): 0.90, 
    ("PPC-CALCINED", 0): 0.95, 
    ("PSC", 0): 0.90,
}

# Standard Deviation (S) for various grades, based on IS 456 / IS 10262 Table 2
def get_standard_deviation(fck_mpa: float) -> float:
    """Returns Standard Deviation S (MPa) based on Characteristic Strength fck."""
    if fck_mpa >= 50:
        return 5.5
    if fck_mpa >= 30:
        return 5.0
    if fck_mpa >= 25:
        return 4.0
    return 3.5 

# -------------------------
# Helper functions
# -------------------------
def classify_workability(slump_mm: float) -> str:
    """Classifies concrete workability based on slump value."""
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
    f'ck is the greater of: fck + 1.65 * S and fck + 6.5 MPa.
    """
    S = get_standard_deviation(fck_mpa)
    X = 6.5 
    
    val_a = fck_mpa + 1.65 * S
    val_b = fck_mpa + X
    
    return max(val_a, val_b)

def predict_strength_adjusted(fck_target: float, w_c_ratio: float, cement_type: str, cement_grade: int) -> float:
    """
    Predicts 28-day strength using IS 10262 target adjusted by cement factor.
    """
    
    # 1. Base Target Strength (IS 10262)
    fcm_is_target = target_mean_strength_is10262(fck_target)

    # 2. Determine factor based on cement type/grade
    cement_key = (cement_type.upper(), cement_grade)
    
    # For non-OPC types, the grade is 0 in the lookup
    if cement_type.upper() not in ["OPC"]:
         cement_key = (cement_type.upper(), 0)

    cement_factor = STRENGTH_FACTOR.get(cement_key, 1.0)
    
    # 3. Apply factor to IS 10262 Target Strength
    predicted_strength = fcm_is_target * cement_factor
    
    # Ensure prediction is reasonable
    predicted_strength = max(fck_target, min(predicted_strength, 120.0))

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
    placing_method: str
):
    """
    Core IS10262-driven (simplified) calculation for one mix.
    """

    # Validate water table availability
    if max_agg_size_mm not in WATER_CONTENT_50MM:
        raise ValueError(f"max_agg_size_mm {max_agg_size_mm} not supported. Supported: {list(WATER_CONTENT_50MM.keys())}")
    
    warnings = []

    # 0) Workability Validation based on Placing Method (IS 10262 Table 4 guidance)
    min_slump, max_slump = PLACING_SLUMP_RANGES.get(placing_method, (75, 125))
    if slump_mm < min_slump or slump_mm > max_slump:
         warnings.append(f"Selected slump of {slump_mm} mm is outside the recommended range ({min_slump}-{max_slump} mm) for '{placing_method.replace('_', ' ').title()}'.")


    # 1) preliminary w/c 
    w_c = w_c_ratio

    # 2) water content for 50 mm slump
    water_50mm = WATER_CONTENT_50MM[max_agg_size_mm]

    # 3) slump adjustment
    delta_slump = slump_mm - 50.0
    correction_pct = (delta_slump / 25.0) * SLUMP_CORRECTION_PERCENT_PER_25MM
    water_required = water_50mm * (1.0 + correction_pct / 100.0)

    # 4) SP reduction
    if use_superplasticizer:
        water_required *= (1.0 - DEFAULT_SP_REDUCTION_PCT / 100.0)

    # 5) cementitious total
    cementitious_total = water_required / w_c

    # 6) split replacements (fractions of cementitious_total)
    scm_masses = {}
    total_scm_fraction = 0.0
    if replacements:
        for k, frac in replacements.items():
            if frac < 0 or frac > 0.9:
                raise ValueError("Replacement fractions must be between 0 and 0.9")
            scm_masses[k] = cementitious_total * frac
            total_scm_fraction += frac
    
    if total_scm_fraction >= 1.0:
        raise ValueError("Total SCM fraction equals or exceeds 1.0. Check inputs.")

    cement_actual = cementitious_total * (1.0 - total_scm_fraction)

    # 7) IS-456 enforcement (min cement & max w/c)
    limits = IS456_DURABILITY_LIMITS_REINFORCED.get(exposure, IS456_DURABILITY_LIMITS_REINFORCED["moderate"])
    min_cement_req = limits["min_cement"]
    max_w_c_allowed = limits["max_w_c"]
    compliance_w_c = (w_c <= max_w_c_allowed)
    compliance_cement = (cement_actual >= min_cement_req)

    if not compliance_w_c:
        warnings.append(f"Selected w/c = {w_c:.3f} exceeds IS-456 max {max_w_c_allowed:.3f} for {exposure} exposure.")
    
    # Minimum cement requirement enforcement
    if not compliance_cement:
        warnings.append(f"Cement {cement_actual:.1f} kg/m³ is below IS-456 minimum {min_cement_req} kg/m³ for {exposure}. Cement mass raised to minimum.")
        
        # Recalculate based on minimum cement requirement
        cement_actual = float(min_cement_req)
        
        # Recalculate cementitious_total based on minimum cement
        cementitious_total = cement_actual / (1.0 - total_scm_fraction)
        
        # Recalculate water_required to maintain the input w/c ratio
        water_required = cementitious_total * w_c
        
        # Re-distribute SCM masses based on new cementitious_total
        for k, frac in replacements.items():
            scm_masses[k] = cementitious_total * frac


    # 8) air fraction
    air_pct = AIR_CONTENT_PCT.get(max_agg_size_mm, 1.0)
    vol_air = air_pct / 100.0

    # 9) volumes
    vol_water = water_required / 1000.0
    vol_cement = cement_actual / (1000.0 * SG["cement"])
    vol_scms = sum([mass / (1000.0 * SG["scm"]) for mass in scm_masses.values()])
    vol_binders = vol_cement + vol_scms

    vol_aggregates = 1.0 - (vol_water + vol_binders + vol_air)
    if vol_aggregates <= 0:
        raise ValueError("Negative aggregate volume - check inputs (cementitious content too high)")

    # 10) aggregate distribution (default fine=40% of aggregate mass)
    fine_vol = 0.40 * vol_aggregates
    coarse_vol = vol_aggregates - fine_vol
    mass_fine = fine_vol * 1000.0 * SG["fine"]
    mass_coarse = coarse_vol * 1000.0 * SG["coarse"]

    # 11) predicted strength (uses IS 10262 target strength adjusted by cement factor)
    predicted_fc_28 = predict_strength_adjusted(fck_mpa, w_c, cement_type, cement_grade)

    # 12) co2 estimates
    ef = EMISSION_FACTORS
    # For Eco-Mix calculation, the base is the full cementitious content assuming only base cement (OPC)
    co2_base = cementitious_total * ef["cement"] + water_required * ef["water"] + mass_fine * ef["sand"] + mass_coarse * ef["coarse"]
    
    # Eco mix CO2 calculation
    co2_eco = cement_actual * ef["cement"] + water_required * ef["water"] + mass_fine * ef["sand"] + mass_coarse * ef["coarse"]
    for scmat, mass in scm_masses.items():
        key = scmat.lower()
        map_keys = {
            "rha":"rha","rice_husk_ash":"rha",
            "ggbs":"ggbs","gGBS":"ggbs",
            "silicafume":"silicafume","silica_fume":"silicafume",
            "metakaolin":"metakaolin",
            "sugarcane_ash": "sugarcane_ash",
            "flyash":"flyash",
        }
        ef_key = map_keys.get(key, key)
        sc_ef = ef.get(ef_key, 0.05)
        co2_eco += mass * sc_ef

    co2_reduction_pct = ((co2_base - co2_eco) / co2_base * 100.0) if co2_base > 0 else 0.0

    # 13) workability
    work_class = classify_workability(slump_mm)

    # 14) durability index (heuristic, simplified)
    if w_c <= 0.40:
        base_dur = 90
    elif w_c <= 0.50:
        base_dur = 75
    elif w_c <= 0.60:
        base_dur = 60
    else:
        base_dur = 45
    exposure_penalty = {"mild": 0, "moderate": -5, "severe": -12, "very_severe": -18, "extreme": -25}
    base_dur += exposure_penalty.get(exposure, -5)
    base_dur += int((cement_type.upper() == "OPC") * -5) 
    base_dur -= int(total_scm_fraction * 20) 
    durability_index = max(20, min(100, base_dur))

    # 15) assemble result
    result = {
        "mix_masses_kg_per_m3": {
            "cement_actual": round(cement_actual, 2),
            **{k: round(v, 2) for k, v in scm_masses.items()},
            "water": round(water_required, 2),
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
            "warnings": warnings
        }
    }

    return result

# -------------------------
# FastAPI + Models
# -------------------------
app = FastAPI(title="EcoMix Design API", version="1.0")

# Allow CORS for frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StandardInput(BaseModel):
    fck_mpa: confloat(gt=0)
    cement_type: str = Field("OPC", description="e.g., OPC, PPC-FLYASH, PSC")
    cement_grade: int = Field(43, description="e.g., 43 or 53 (0 for PPC/PSC)")
    w_c_ratio: confloat(gt=0)
    slump_mm: confloat(ge=0)
    exposure: str = Field("moderate")
    max_agg_size_mm: int = Field(20)
    placing_method: str = Field("manual_placing")
    mix_proportions: Optional[Dict[str, float]] = Field(None, description="Optional manual mix proportions (cement, fine, coarse)")
    admixtures: Optional[Dict[str, float]] = Field(None, description="Admixture name -> dosage%")
    use_superplasticizer: Optional[bool] = False

class EcoInput(BaseModel):
    fck_mpa: confloat(gt=0)
    cement_type: str = Field("OPC")
    cement_grade: int = Field(43)
    w_c_ratio: confloat(gt=0)
    slump_mm: confloat(ge=0)
    exposure: str = Field("moderate")
    max_agg_size_mm: int = Field(20)
    placing_method: str = Field("manual_placing")
    admixtures: Optional[Dict[str, float]] = Field(None)
    use_superplasticizer: Optional[bool] = False
    # SCM fractions (fractions of cementitious total, max 0.9 combined)
    rice_husk_ash_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    coconut_husk_ash_frac: Optional[confloat(ge=0, le=1.0)] = 0.0
    sugarcane_ash_frac: Optional[confloat(ge=0, le=0.2)] = 0.0 # Max 20% for Sugarcane Ash
    flyash_frac: Optional[confloat(ge=0, le=0.6)] = 0.0
    ggbs_frac: Optional[confloat(ge=0, le=0.9)] = 0.0
    silica_fume_frac: Optional[confloat(ge=0, le=0.1)] = 0.0
    metakaolin_frac: Optional[confloat(ge=0, le=0.1)] = 0.0 # Max 10% for Metakaolin

class CompareInput(BaseModel):
    standard: StandardInput
    eco: EcoInput

# Endpoints
@app.post("/api/design/standard")
def design_standard(payload: StandardInput):
    try:
        replacements = {}
        # Standard mode: no SCM replacements by default 
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
            placing_method=payload.placing_method
        )
        # Add inputs echo
        res["inputs"] = payload.model_dump() 
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/design/eco")
def design_eco(payload: EcoInput):
    try:
        replacements = {
            "rha": float(payload.rice_husk_ash_frac or 0.0),
            "coconut_ash": float(payload.coconut_husk_ash_frac or 0.0),
            "sugarcane_ash": float(payload.sugarcane_ash_frac or 0.0),
            "flyash": float(payload.flyash_frac or 0.0),
            "ggbs": float(payload.ggbs_frac or 0.0),
            "silicafume": float(payload.silica_fume_frac or 0.0),
            "metakaolin": float(payload.metakaolin_frac or 0.0),
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
            placing_method=payload.placing_method
        )
        res["inputs"] = payload.model_dump()
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/design/compare")
def design_compare(payload: CompareInput):
    try:
        # --- Standard Mix Calculation ---
        st = payload.standard
        res_standard = compute_mix_from_inputs(
            fck_mpa=st.fck_mpa,
            w_c_ratio=float(st.w_c_ratio),
            slump_mm=float(st.slump_mm),
            max_agg_size_mm=st.max_agg_size_mm,
            replacements={},  # none
            use_superplasticizer=st.use_superplasticizer,
            exposure=st.exposure,
            cement_type=st.cement_type,
            cement_grade=st.cement_grade,
            placing_method=st.placing_method
        )
        
        # --- Eco Mix Calculation ---
        eco = payload.eco
        replacements = {
            "rha": float(eco.rice_husk_ash_frac or 0.0),
            "coconut_ash": float(eco.coconut_husk_ash_frac or 0.0),
            "sugarcane_ash": float(eco.sugarcane_ash_frac or 0.0),
            "flyash": float(eco.flyash_frac or 0.0),
            "ggbs": float(eco.ggbs_frac or 0.0),
            "silicafume": float(eco.silica_fume_frac or 0.0),
            "metakaolin": float(eco.metakaolin_frac or 0.0),
        }
        replacements = {k: v for k, v in replacements.items() if v and v > 0.0}
        
        res_eco = compute_mix_from_inputs(
            fck_mpa=eco.fck_mpa,
            w_c_ratio=float(eco.w_c_ratio),
            slump_mm=float(eco.slump_mm),
            max_agg_size_mm=eco.max_agg_size_mm,
            replacements=replacements,
            use_superplasticizer=eco.use_superplasticizer,
            exposure=eco.exposure,
            cement_type=eco.cement_type,
            cement_grade=eco.cement_grade,
            placing_method=eco.placing_method
        )

        # --- Compute Comparisons ---
        co2_base = res_standard["co2"]["co2_base_kg_per_m3"]
        co2_eco = res_eco["co2"]["co2_eco_kg_per_m3"]
        co2_reduction_pct = round(((co2_base - co2_eco) / co2_base * 100.0) if co2_base > 0 else 0.0, 2)
        
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
