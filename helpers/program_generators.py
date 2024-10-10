import random

# Function to generate program code for program 0, with optional arguments for lp_cut and hp_cut
def generate_program_0(lp_cut_range, hp_cut_range, lp_cut=None, hp_cut=None):
    lp_cut_min, lp_cut_max = lp_cut_range
    hp_cut_min, hp_cut_max = hp_cut_range
    
    # Use provided values or generate randomly
    LP_CUT = lp_cut if lp_cut is not None else random.randint(lp_cut_min, lp_cut_max)
    HP_CUT = hp_cut if hp_cut is not None else random.randint(hp_cut_min, hp_cut_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'lp_cut = hslider("lp_cut", {LP_CUT}, {LP_CUT_MIN}, {LP_CUT_MAX}, 1);\n'
        'hp_cut = hslider("hp_cut", {HP_CUT}, {HP_CUT_MIN}, {HP_CUT_MAX}, 1);\n'
        'process = no.noise:fi.lowpass(3, lp_cut):fi.highpass(10, hp_cut);'
    )

    formatted_code = program_code.format(
        LP_CUT=LP_CUT,
        LP_CUT_MIN=lp_cut_min,
        LP_CUT_MAX=lp_cut_max,
        HP_CUT=HP_CUT,
        HP_CUT_MIN=hp_cut_min,
        HP_CUT_MAX=hp_cut_max
    )

    return formatted_code, LP_CUT, HP_CUT

# Function to generate program code for program 1, with optional arguments for saw_freq and sine_freq
def generate_program_1(saw_freq_range, sine_freq_range, saw_freq=None, sine_freq=None):
    saw_freq_min, saw_freq_max = saw_freq_range
    sine_freq_min, sine_freq_max = sine_freq_range
    
    # Use provided values or generate randomly
    SAW_FREQ = saw_freq if saw_freq is not None else random.randint(saw_freq_min, saw_freq_max)
    SINE_FREQ = sine_freq if sine_freq is not None else random.randint(sine_freq_min, sine_freq_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'saw_freq = hslider("saw_freq", {SAW_FREQ}, {SAW_FREQ_MIN}, {SAW_FREQ_MAX}, 1);\n'
        'sine_freq = hslider("sine_freq", {SINE_FREQ}, {SINE_FREQ_MIN}, {SINE_FREQ_MAX}, 1);\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac;\n'
        'process = sineOsc(sine_freq) + sawOsc(saw_freq);'
    )

    formatted_code = program_code.format(
        SAW_FREQ=SAW_FREQ,
        SAW_FREQ_MIN=saw_freq_min,
        SAW_FREQ_MAX=saw_freq_max,
        SINE_FREQ=SINE_FREQ,
        SINE_FREQ_MIN=sine_freq_min,
        SINE_FREQ_MAX=sine_freq_max
    )

    return formatted_code, SAW_FREQ, SINE_FREQ

# Function to generate program code for program 2, with optional arguments for amp and carrier
def generate_program_2(amp_range, carrier_range, amp=None, carrier=None):
    amp_min, amp_max = amp_range
    carrier_min, carrier_max = carrier_range
    
    # Use provided values or generate randomly
    AMP = amp if amp is not None else random.uniform(amp_min, amp_max)
    CARRIER = carrier if carrier is not None else random.uniform(carrier_min, carrier_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'amp = hslider("amp", {AMP}, {AMP_MIN}, {AMP_MAX}, 0.01);\n'
        'carrier = hslider("carrier", {CARRIER}, {CARRIER_MIN}, {CARRIER_MAX}, 0.01);\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;\n'
        'process = no.noise * sineOsc(carrier) * amp;'
    )

    formatted_code = program_code.format(
        AMP=AMP,
        AMP_MIN=amp_min,
        AMP_MAX=amp_max,
        CARRIER=CARRIER,
        CARRIER_MIN=carrier_min,
        CARRIER_MAX=carrier_max
    )

    return formatted_code, AMP, CARRIER

# Function to generate program code for program 3, with optional arguments for amp and carrier
def generate_program_3(amp_range, carrier_range, amp=None, carrier=None):
    amp_min, amp_max = amp_range
    carrier_min, carrier_max = carrier_range
    
    # Use provided values or generate randomly
    AMP = amp if amp is not None else random.uniform(amp_min, amp_max)
    CARRIER = carrier if carrier is not None else random.randint(carrier_min, carrier_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'carrier = hslider("carrier", {CARRIER}, {CARRIER_MIN}, {CARRIER_MAX}, 1);\n'
        'amp = hslider("amp", {AMP}, {AMP_MIN}, {AMP_MAX}, 1);\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac;\n'
        'process = sineOsc(amp) * sawOsc(carrier);'
    )

    formatted_code = program_code.format(
        CARRIER=CARRIER,
        CARRIER_MIN=carrier_min,
        CARRIER_MAX=carrier_max,
        AMP=AMP,
        AMP_MIN=amp_min,
        AMP_MAX=amp_max
    )

    return formatted_code, AMP, CARRIER

# Program selection function
def choose_program(program_id, var1_range, var2_range, var1=None, var2=None):
    if program_id == 0:
        return generate_program_0(var1_range, var2_range, var1, var2)
    elif program_id == 1:
        return generate_program_1(var1_range, var2_range, var1, var2)
    elif program_id == 2:
        return generate_program_2(var1_range, var2_range, var1, var2)
    elif program_id == 3:
        return generate_program_3(var1_range, var2_range, var1, var2)
    else:
        raise ValueError("Invalid program ID")

