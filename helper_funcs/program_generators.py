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
        'process = no.noise:fi.lowpass(5, lp_cut):fi.highpass(5, hp_cut);'
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

# alternative code for program 0, using saw instead of noise with optional arguments for lp_cut and hp_cut
def generate_program_0_v1(lp_cut_range, hp_cut_range, lp_cut=None, hp_cut=None):
    lp_cut_min, lp_cut_max = lp_cut_range
    hp_cut_min, hp_cut_max = hp_cut_range
    
    # Use provided values or generate randomly
    LP_CUT = lp_cut if lp_cut is not None else random.randint(lp_cut_min, lp_cut_max)
    HP_CUT = hp_cut if hp_cut is not None else random.randint(hp_cut_min, hp_cut_max)
    SAW_FREQ = 30
    program_code = (
        'import("stdfaust.lib");\n'
        'lp_cut = hslider("lp_cut", {LP_CUT}, {LP_CUT_MIN}, {LP_CUT_MAX}, 1);\n'
        'hp_cut = hslider("hp_cut", {HP_CUT}, {HP_CUT_MIN}, {HP_CUT_MAX}, 1);\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac;\n'
        'process = sawOsc({SAW_FREQ}):fi.lowpass(5, lp_cut):fi.highpass(5, hp_cut);'
    )

    formatted_code = program_code.format(
        LP_CUT=LP_CUT,
        LP_CUT_MIN=lp_cut_min,
        LP_CUT_MAX=lp_cut_max,
        HP_CUT=HP_CUT,
        HP_CUT_MIN=hp_cut_min,
        HP_CUT_MAX=hp_cut_max,
        SAW_FREQ=SAW_FREQ,
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

def generate_program_3_variation(amp_range, carrier_range, amp=None, carrier=None):
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
        'process = sineOsc(amp) * sineOsc(carrier);'
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


# program 0_1D, with optional arguments for lp_cut and hp_cut
def generate_program_0_1D(lp_cut_range, lp_cut=None, hp_cut=None):
    lp_cut_min, lp_cut_max = lp_cut_range
    
    # Use provided values or generate randomly
    LP_CUT = lp_cut if lp_cut is not None else random.randint(lp_cut_min, lp_cut_max)
    HP_CUT = hp_cut if hp_cut is not None else 1000

    program_code = (
        'import("stdfaust.lib");\n'
        'lp_cut = hslider("lp_cut", {LP_CUT}, {LP_CUT_MIN}, {LP_CUT_MAX}, 1);\n'
        'process = no.noise:fi.lowpass(3, lp_cut):fi.highpass(10, hp_cut);'
    )

    formatted_code = program_code.format(
        LP_CUT=LP_CUT,
        LP_CUT_MIN=lp_cut_min,
        LP_CUT_MAX=lp_cut_max,
        HP_CUT=HP_CUT,
    )

    return formatted_code, LP_CUT, HP_CUT

# Program selection function
def choose_program(program_id, var1_range, var2_range=None, var1=None, var2=None):
    if program_id == 0:
        return generate_program_0(var1_range, var2_range, var1, var2)
    elif program_id == 1:
        return generate_program_1(var1_range, var2_range, var1, var2)
    elif program_id == 2:
        return generate_program_2(var1_range, var2_range, var1, var2)
    elif program_id == 3:
        return generate_program_3(var1_range, var2_range, var1, var2)
    elif program_id == 0.1:
        return generate_program_0_1D(var1_range,var1,var2)
    else:
        raise ValueError("Invalid program ID")

# Param ranges
def generate_parameters(program_id):
    if program_id == 0:
        var1_range = (50, 1000)
        var2_range = (1, 120)
    elif program_id == 1:
        var1_range = (30, 1000)
        var2_range = (30, 1000)
    elif program_id == 2:
        var1_range = (0.1, 1)
        var2_range = (1, 20)
    elif program_id == 3:
        var1_range = (1, 20)
        var2_range = (10, 1000)
    else:
        raise ValueError(f"Unknown program_id: {program_id}")

    var1 = random.uniform(*var1_range)
    var2 = random.uniform(*var2_range)

    if program_id in [0, 1, 3]:
        var1 = int(var1)
    if program_id != 2:
        var2 = int(var2)

    return var1_range, var2_range, var1, var2

# 1 Dimensional program
# Function to generate program code for program 0, with optional argument for lp_cut only
def generate_1_1d(lp_cut_range, lp_cut=None):
    lp_cut_min, lp_cut_max = lp_cut_range

    # Use provided value or generate randomly
    LP_CUT = lp_cut if lp_cut is not None else random.randint(lp_cut_min, lp_cut_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'lp_cut = hslider("lp_cut", {LP_CUT}, {LP_CUT_MIN}, {LP_CUT_MAX}, 1);\n'
        'process = no.noise:fi.highpass(3, lp_cut);'
    )

    formatted_code = program_code.format(
        LP_CUT=LP_CUT,
        LP_CUT_MIN=lp_cut_min,
        LP_CUT_MAX=lp_cut_max
    )

    return formatted_code, LP_CUT

def generate_2_1d(carrier_range, carrier=None):
    carrier_min, carrier_max = carrier_range

    # Use provided value or generate randomly
    CARRIER = carrier if carrier is not None else random.uniform(carrier_min, carrier_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'carrier = hslider("carrier", {CARRIER}, {CARRIER_MIN}, {CARRIER_MAX}, 0.01);\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;\n'
        'process = no.noise * sineOsc(carrier);'
    )

    formatted_code = program_code.format(
        CARRIER=CARRIER,
        CARRIER_MIN=carrier_min,
        CARRIER_MAX=carrier_max
    )

    return formatted_code, CARRIER

def generate_program_chirp_nodelay(increase_range, pitch_range, increase_default=None, pitch_default=None):
    increase_min, increase_max = increase_range
    pitch_min, pitch_max = pitch_range

    # Random defaults if not provided
    INCREASE_DEFAULT = increase_default if increase_default is not None else round(random.uniform(increase_min, increase_max), 2)
    PITCH_DEFAULT = pitch_default if pitch_default is not None else round(random.uniform(pitch_min, pitch_max), 2)

    program_code = (
        'import("stdfaust.lib");\n'
        'increase_speed = hslider("increase_speed", {INCREASE_DEFAULT}, {INCREASE_MIN}, {INCREASE_MAX}, 0.1);\n'
        'starting_pitch = hslider("starting_pitch", {PITCH_DEFAULT}, {PITCH_MIN}, {PITCH_MAX}, 0.1);\n\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac : *(3*ma.PI) : sin;\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac : +(0.5);\n'
        'increasing_pitch(rate) = _ ~ +(rate/ma.SR) : exp;\n\n'
        'process = sineOsc(increasing_pitch(increase_speed) + starting_pitch);'
    )

    formatted_code = program_code.format(
        INCREASE_DEFAULT=INCREASE_DEFAULT,
        INCREASE_MIN=increase_min,
        INCREASE_MAX=increase_max,
        PITCH_DEFAULT=PITCH_DEFAULT,
        PITCH_MIN=pitch_min,
        PITCH_MAX=pitch_max
    )

    return formatted_code, INCREASE_DEFAULT, PITCH_DEFAULT

# out-of-domain
def generate_program_chirplet_pulse(increase_rate_range, pulse_rate_range, mult_range=(50, 1000), increase_default=None, pulse_default=None, mult_value=None):
    # out of domain 
    increase_min, increase_max = increase_rate_range
    pulse_min, pulse_max = pulse_rate_range
    mult_min, mult_max = mult_range

    # Choose default values if not provided
    INCREASE_DEFAULT = increase_default if increase_default is not None else round(random.uniform(increase_min, increase_max), 2)
    PULSE_DEFAULT = pulse_default if pulse_default is not None else round(random.uniform(pulse_min, pulse_max), 2)
    MULT = mult_value if mult_value is not None else random.randint(mult_min, mult_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'increase_speed = hslider("increase_speed", {INCREASE_DEFAULT}, {INCREASE_MIN}, {INCREASE_MAX}, 0.1);\n'
        'pulse_rate = hslider("pulse_rate", {PULSE_DEFAULT}, {PULSE_MIN}, {PULSE_MAX}, 0.1);\n\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac : *(3*ma.PI) : sin;\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac : +(0.5);\n'
        'increasing_pitch(rate) = _ ~ +(rate/ma.SR) : exp;\n\n'
        'process = sineOsc(30 + increasing_pitch(increase_speed) + sawOsc(pulse_rate) * {MULT});'
    )

    formatted_code = program_code.format(
        INCREASE_DEFAULT=INCREASE_DEFAULT,
        INCREASE_MIN=increase_min,
        INCREASE_MAX=increase_max,
        PULSE_DEFAULT=PULSE_DEFAULT,
        PULSE_MIN=pulse_min,
        PULSE_MAX=pulse_max,
        MULT=MULT
    )

    return formatted_code, INCREASE_DEFAULT, PULSE_DEFAULT


def generate_program_delayed_chirplet(increase_range, pitch_range, delay_range=(5000, 30000), increase_default=None, pitch_default=None, delay_value=None):
    increase_min, increase_max = increase_range
    pitch_min, pitch_max = pitch_range
    delay_min, delay_max = delay_range

    # Random defaults if not provided
    INCREASE_DEFAULT = increase_default if increase_default is not None else round(random.uniform(increase_min, increase_max), 2)
    PITCH_DEFAULT = pitch_default if pitch_default is not None else round(random.uniform(pitch_min, pitch_max), 2)
    RAND_START = delay_value if delay_value is not None else random.randint(delay_min, delay_max)

    program_code = (
        'import("stdfaust.lib");\n'
        'increase_speed = hslider("increase_speed", {INCREASE_DEFAULT}, {INCREASE_MIN}, {INCREASE_MAX}, 0.1);\n'
        'starting_pitch = hslider("starting_pitch", {PITCH_DEFAULT}, {PITCH_MIN}, {PITCH_MAX}, 0.1);\n\n'
        'sineOsc(f) = +(f/ma.SR) ~ ma.frac : *(3*ma.PI) : sin;\n'
        'sawOsc(f) = +(f/ma.SR) ~ ma.frac : +(0.5);\n'
        'increasing_pitch(rate) = _ ~ +(rate/ma.SR) : exp;\n\n'
        'process = sineOsc(increasing_pitch(increase_speed) : de.delay({RAND_START}, 48000) + starting_pitch);'
    )

    formatted_code = program_code.format(
        INCREASE_DEFAULT=INCREASE_DEFAULT,
        INCREASE_MIN=increase_min,
        INCREASE_MAX=increase_max,
        PITCH_DEFAULT=PITCH_DEFAULT,
        PITCH_MIN=pitch_min,
        PITCH_MAX=pitch_max,
        RAND_START=RAND_START
    )

    return formatted_code, INCREASE_DEFAULT, PITCH_DEFAULT

