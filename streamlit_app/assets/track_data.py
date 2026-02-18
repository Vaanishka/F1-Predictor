# streamlit_app/assets/track_data.py

TRACK_INFO = {
    # --- ROUND 1: BAHRAIN ---
    "Bahrain Grand Prix": {
        "length_speed": {
            "title": "Stop-Go Power Circuit",
            "desc": "A traction-dominated circuit defined by four long straights and heavy braking zones. Lap time is found in the exit of slow corners (Turns 1, 4, and 10), demanding a setup that prioritizes longitudinal grip over high-speed stability."
        },
        "corners": {
            "title": "Point-and-Shoot",
            "desc": "Turn 10 is the technical highlight—a blind, off-camber left-hander that invites front-locking. The high-speed Turn 6-7 complex is the only section demanding significant aerodynamic load."
        },
        "evolution": {
            "title": "High Evolution / Abrasive",
            "desc": "The granite-based asphalt is extremely abrasive. Track evolution is high in FP1 as sand is cleaned off, but the surface remains aggressive on rear tyres, usually dictating a multi-stop strategy."
        },
        "drs": {
            "title": "3 Powerful Zones",
            "desc": "With three DRS zones, overtaking is highly possible. The run to Turn 1 is the prime spot; a car with efficient DRS can shed massive drag here."
        }
    },

    # --- ROUND 2: SAUDI ARABIA ---
    "Saudi Arabian Grand Prix": {
        "length_speed": {
            "title": "Fastest Street Circuit",
            "desc": "Jeddah defies convention with an average speed over 250km/h. It requires a low-drag setup similar to Monza but with the walls of Monaco. Precision at high speed is the ultimate differentiator."
        },
        "corners": {
            "title": "High-Speed Flow",
            "desc": "A relentless series of high-speed sweeps. Turns 22-24 require total commitment. Unlike Bahrain, there is almost no heavy braking; the lap is about carrying momentum."
        },
        "evolution": {
            "title": "Low Degradation",
            "desc": "The surface is remarkably smooth and offers high grip immediately. Tyre wear is low, often allowing for a comfortable one-stop strategy."
        },
        "drs": {
            "title": "Tactical Games",
            "desc": "The DRS detection point for the final corner is infamous for 'DRS Chicken' games, where drivers brake early to get the advantage on the main straight."
        }
    },

    # --- ROUND 3: AUSTRALIA ---
    "Australian Grand Prix": {
        "length_speed": {
            "title": "High-Speed Park",
            "desc": "Following the 2022 layout changes, Albert Park is now a much faster flowing circuit. The removal of the old chicane has created a massive full-throttle zone along the lake."
        },
        "corners": {
            "title": "Flow & Precision",
            "desc": "The high-speed chicane at Turns 9-10 is the critical test of car stability. Drivers must attack the kerbs aggressively without unsettling the car."
        },
        "evolution": {
            "title": "Significant Ramp-Up",
            "desc": "Being a semi-permanent street circuit, the track is very 'green' and slippery on Friday but evolves rapidly as rubber is laid down."
        },
        "drs": {
            "title": "4 DRS Zones",
            "desc": "Features four DRS zones, designed to keep cars close. However, passing remains difficult due to the narrow racing line and dirty air in the fast sections."
        }
    },

    # --- ROUND 4: JAPAN ---
    "Japanese Grand Prix": {
        "length_speed": {
            "title": "The Driver's Benchmark",
            "desc": "The ultimate test of man and machine. A figure-of-eight layout that punishes weakness. Aerodynamic efficiency and chassis balance are more important here than anywhere else."
        },
        "corners": {
            "title": "The Esses (Sector 1)",
            "desc": "The iconic 'S' Curves demand a rhythmic flow; one mistake in the first corner compromises the entire sector. 130R is a fearless flat-out blast."
        },
        "evolution": {
            "title": "High Tyre Stress",
            "desc": "The high lateral loads through the fast corners put immense energy into the tyres. Thermal degradation is the limiting factor, not wear."
        },
        "drs": {
            "title": "Limited Effect",
            "desc": "Only one DRS zone on the main straight. Overtaking usually happens into the Casio Triangle chicane or through sheer bravery into Turn 1."
        }
    },

    # --- ROUND 5: CHINA ---
    "Chinese Grand Prix": {
        "length_speed": {
            "title": "Front-Limited Circuit",
            "desc": "Dominated by the massive 1.2km back straight. However, the lap time is dictated by the never-ending Turn 1 and Turn 13, which destroy front-left tyres."
        },
        "corners": {
            "title": "The Snail",
            "desc": "Turn 1 represents a 270-degree tightening spiral that requires patience. Entering too fast ruins the exit and kills the tires instantly."
        },
        "evolution": {
            "title": "Graining Risk",
            "desc": "Cooler temperatures often lead to front graining. Setup is a compromise between top speed for the straight and downforce to protect the front tires."
        },
        "drs": {
            "title": "The Back Straight",
            "desc": "The DRS zone on the back straight is one of the most powerful on the calendar, making passing into the hairpin relatively standard."
        }
    },

    # --- ROUND 6: MIAMI ---
    "Miami Grand Prix": {
        "length_speed": {
            "title": "Hybrid Street Track",
            "desc": "Combines a tight, technical street section with long, high-speed straights. Setup is tricky: you need low drag for speed but soft suspension for the kerbs."
        },
        "corners": {
            "title": "The Tight Chicane",
            "desc": "The Turn 14-15 chicane is incredibly slow, uphill, and blind. It breaks the rhythm and requires excellent mechanical grip and traction."
        },
        "evolution": {
            "title": "Surface Grip Issues",
            "desc": "The track is known for low grip off-line. Going wide usually means picking up dirt and losing seconds, making overtaking risky."
        },
        "drs": {
            "title": "Triple Threat",
            "desc": "Three long DRS zones help keep the racing close, but the dirty air in the tight stadium section can make following difficult before the straight."
        }
    },

    # --- ROUND 7: IMOLA (Emilia Romagna) ---
    "Emilia Romagna Grand Prix": {
        "length_speed": {
            "title": "Old School Flow",
            "desc": "Anti-clockwise, narrow, and bumpy. Imola punishes mistakes with gravel traps instead of runoff. It rewards a car that can ride kerbs aggressively."
        },
        "corners": {
            "title": "Acque Minerali",
            "desc": "A stunning downhill-uphill complex. Drivers brake while turning and loading the suspension, making it a prime spot for spins."
        },
        "evolution": {
            "title": "Narrow Line",
            "desc": "Overtaking is notoriously difficult. Track position is king. Strategy often focuses on the 'undercut' rather than on-track passing."
        },
        "drs": {
            "title": "Single Zone",
            "desc": "Only one DRS zone on the main straight. The run to Tamburello is the only realistic overtaking opportunity."
        }
    },

    # --- ROUND 8: MONACO ---
    "Monaco Grand Prix": {
        "length_speed": {
            "title": "The Crown Jewel",
            "desc": "The slowest average speed of the year. Aerodynamics are irrelevant for drag; teams run 'barn door' wings for maximum downforce. Qualifying is 99% of the race."
        },
        "corners": {
            "title": "Loews Hairpin",
            "desc": "The slowest corner in F1 (approx 45km/h). Teams modify steering racks just to make the turn radius. Precision > Power."
        },
        "evolution": {
            "title": "Massive Evolution",
            "desc": "The track is public road. It starts dusty and evolves seconds per lap by Saturday. Confidence near the barriers is the only metric that matters."
        },
        "drs": {
            "title": "Irrelevant",
            "desc": "DRS is barely useful here. The straights are too short and curved. Overtaking requires a massive pace advantage or a mistake from ahead."
        }
    },

    # --- ROUND 9: CANADA ---
    "Canadian Grand Prix": {
        "length_speed": {
            "title": "Stop-Go Classic",
            "desc": "A high-speed circuit broken up by chicanes. It is the toughest test for brakes on the calendar. Brake cooling and stability are critical."
        },
        "corners": {
            "title": "Wall of Champions",
            "desc": "The final chicane invites drivers to use every inch of kerb. Get it wrong, and the famous wall on the exit ends your race instantly."
        },
        "evolution": {
            "title": "Green Track",
            "desc": "Located on an island park, the surface is often dirty on Friday. Rain is a frequent guest, adding chaos to the mix."
        },
        "drs": {
            "title": "Slingshot",
            "desc": "The DRS zone before the final chicane allows cars to close up, setting up a pass either into the chicane or on the main straight."
        }
    },

    # --- ROUND 10: SPAIN ---
    "Spanish Grand Prix": {
        "length_speed": {
            "title": "Aerodynamic Test Bench",
            "desc": "If a car is fast here, it is fast everywhere. A perfect mix of high, medium, and low-speed corners that tests every aspect of the chassis."
        },
        "corners": {
            "title": "Turn 3",
            "desc": "A never-ending, high-speed right hander that kills the front-left tyre. It is a pure test of downforce and neck muscles."
        },
        "evolution": {
            "title": "Overheating",
            "desc": "High track temperatures often lead to thermal degradation. Management of the rear tyres in the final sector is key to a good lap."
        },
        "drs": {
            "title": "Main Straight",
            "desc": "The removal of the final chicane has improved flow onto the main straight, making DRS more effective into Turn 1."
        }
    },

    # --- ROUND 11: AUSTRIA ---
    "Austrian Grand Prix": {
        "length_speed": {
            "title": "The Sprint",
            "desc": "Shortest lap time on the calendar (approx 63s). The field is always tightly packed. Power-sensitive due to the steep uphill climbs."
        },
        "corners": {
            "title": "Off-Camber Exits",
            "desc": "Turns 3 and 4 are uphill braking zones with tricky downhill exits. Running wide often results in track limits penalties."
        },
        "evolution": {
            "title": "Kerb Risk",
            "desc": "The aggressive 'sausage kerbs' can destroy floors and suspension. Drivers must be precise to avoid mechanical damage."
        },
        "drs": {
            "title": "Triple Zone",
            "desc": "Three consecutive DRS zones allow a car to pass, get passed back, and pass again. It creates frantic racing."
        }
    },

    # --- ROUND 12: BRITAIN (Silverstone) ---
    "British Grand Prix": {
        "length_speed": {
            "title": "Home of F1",
            "desc": "A power and aero temple. The high-speed nature puts the highest lateral load of the year on the tyres. Punctures are a historic risk here."
        },
        "corners": {
            "title": "Maggotts-Becketts",
            "desc": "The greatest sequence in motorsport. A left-right-left-right sweep taken at over 290km/h. It separates the elite chassis from the rest."
        },
        "evolution": {
            "title": "Wind Sensitivity",
            "desc": "The flat airfield location means wind direction changes car balance drastically from corner to corner."
        },
        "drs": {
            "title": "Hangar Straight",
            "desc": "The run down Hangar Straight is the primary overtaking spot, often set up by a good exit from Chapel."
        }
    },

    # --- ROUND 13: HUNGARY ---
    "Hungarian Grand Prix": {
        "length_speed": {
            "title": "Monaco without Walls",
            "desc": "Tight, twisty, and relentless. There is no time to rest. High downforce is required, and cooling is often an issue in the summer heat."
        },
        "corners": {
            "title": "Technical Flow",
            "desc": "Sector 2 is a constant dance of steering input. One missed apex ruins the rhythm for the next four corners."
        },
        "evolution": {
            "title": "Dusty Surface",
            "desc": "Often very dusty on Friday. Overtaking is notoriously hard, earning it the nickname 'Monaco without the glamour'."
        },
        "drs": {
            "title": "Main Straight Only",
            "desc": "The only real passing spot is Turn 1. A good exit from the final corner is essential to have a chance."
        }
    },

    # --- ROUND 14: BELGIUM (Spa) ---
    "Belgian Grand Prix": {
        "length_speed": {
            "title": "The Giant",
            "desc": "The longest track on the calendar (7km). It features massive elevation changes and micro-climates where it rains on one part and is dry on another."
        },
        "corners": {
            "title": "Eau Rouge / Raidillon",
            "desc": "The most famous corner in the world. A blind, uphill, full-throttle compression that tests bravery and engine power."
        },
        "evolution": {
            "title": "Variable Grip",
            "desc": "Setup is a nightmare compromise: low drag for Sector 1/3, high downforce for the technical Sector 2."
        },
        "drs": {
            "title": "Kemmel Straight",
            "desc": "The long run after Raidillon is the easiest passing spot on the calendar if you have the tow."
        }
    },

    # --- ROUND 15: DUTCH (Zandvoort) ---
    "Dutch Grand Prix": {
        "length_speed": {
            "title": "The Rollercoaster",
            "desc": "Old school, narrow, and banked. Zandvoort flows through the sand dunes with zero room for error. It feels incredibly fast from the cockpit."
        },
        "corners": {
            "title": "The Banking",
            "desc": "Turn 3 (Hugenholtz) and the final corner (Luyendyk) feature steep banking (18 degrees), allowing unique lines and higher speeds."
        },
        "evolution": {
            "title": "Sand & Wind",
            "desc": "Wind blows sand onto the track, constantly changing grip levels. High downforce is required to stick the car to the banking."
        },
        "drs": {
            "title": "Banked DRS",
            "desc": "DRS can be opened early through the final banked corner, creating a slingshot effect onto the main straight."
        }
    },

    # --- ROUND 16: ITALY (Monza) ---
    "Italian Grand Prix": {
        "length_speed": {
            "title": "Temple of Speed",
            "desc": "The fastest track in history. Teams bring special 'skinny' wings found nowhere else. Brakes run cold on straights then face massive shock loads."
        },
        "corners": {
            "title": "The Parabolica",
            "desc": "A long, accelerating right-hander that determines your speed down the main straight. Track limits on the exit are a constant issue."
        },
        "evolution": {
            "title": "Slipstream City",
            "desc": "Qualifying is a tactical mess of cars trying to get a 'tow'. Running in clean air is significantly slower here."
        },
        "drs": {
            "title": "Potent",
            "desc": "DRS is less effective than usual because the rear wings are already so small, but the slipstream effect is massive."
        }
    },

    # --- ROUND 17: AZERBAIJAN (Baku) ---
    "Azerbaijan Grand Prix": {
        "length_speed": {
            "title": "City of Speed",
            "desc": "A paradox: The longest straight on the calendar combined with the tightest castle section. Setup is a nightmare compromise."
        },
        "corners": {
            "title": "Castle Section",
            "desc": "Turn 8-12 winds through the medieval city walls. The track is barely wider than the car. A mistake here causes a guaranteed red flag."
        },
        "evolution": {
            "title": "Low Grip",
            "desc": "The street surface is very smooth and low grip. Tyre warm-up is a major issue before the restarts."
        },
        "drs": {
            "title": "2.2km Flat Out",
            "desc": "The run from Turn 16 to Turn 1 is effectively a 2.2km straight. The slipstream/DRS effect is enormous."
        }
    },

    # --- ROUND 18: SINGAPORE ---
    "Singapore Grand Prix": {
        "length_speed": {
            "title": "The Physical Test",
            "desc": "Hot, humid, and bumpy. The longest race by duration (often hitting the 2-hour limit). It destroys gearboxes with thousands of shifts."
        },
        "corners": {
            "title": "90-Degree Hell",
            "desc": "A series of 90-degree street corners that require excellent traction. There is no rest for the driver."
        },
        "evolution": {
            "title": "Street Evolution",
            "desc": "Track ramps up massively. The walls are unforgiving. Safety cars are a guaranteed statistical probability."
        },
        "drs": {
            "title": "Short Zones",
            "desc": "DRS zones are short and passing is extremely difficult. Qualifying position is usually race position."
        }
    },

    # --- ROUND 19: USA (COTA) ---
    "United States Grand Prix": {
        "length_speed": {
            "title": "Modern Classic",
            "desc": "Designed to mimic the best sectors of historic tracks. It features the Maggotts complex, the Hockenheim stadium, and Turkey's Turn 8."
        },
        "corners": {
            "title": "Turn 1",
            "desc": "A steep uphill braking zone into a blind apex hairpin. It invites late braking and wide lines, creating a perfect overtaking spot."
        },
        "evolution": {
            "title": "Bumps",
            "desc": "The soft soil causes bumps to form over time, forcing teams to raise ride heights and compromising ground effect downforce."
        },
        "drs": {
            "title": "Back Straight",
            "desc": "The long back straight provides the easiest overtaking opportunity, usually done under braking into Turn 12."
        }
    },

    # --- ROUND 20: MEXICO ---
    "Mexican Grand Prix": {
        "length_speed": {
            "title": "Thin Air",
            "desc": "Racing at 2,200m altitude. The air is 25% thinner, meaning maximum downforce wings produce Monza levels of grip. Cooling is critical."
        },
        "corners": {
            "title": "Stadium Section",
            "desc": "The cars enter a baseball stadium filled with fans. It is very slow and technical, requiring good mechanical grip."
        },
        "evolution": {
            "title": "Low Drag",
            "desc": "Despite big wings, the thin air means drag is very low. Top speeds are some of the highest of the season."
        },
        "drs": {
            "title": "Long Run",
            "desc": "The straight is massive, but cooling issues often prevent cars from following closely enough to use DRS effectively."
        }
    },

    # --- ROUND 21: BRAZIL (Interlagos) ---
    "São Paulo Grand Prix": {
        "length_speed": {
            "title": "Interlagos Magic",
            "desc": "A short, anti-clockwise bowl that always produces drama. Weather can change in minutes. It flows beautifully and encourages racing."
        },
        "corners": {
            "title": "Senna 'S'",
            "desc": "A downhill, off-camber plunge into Turn 1. It is easy to lock a wheel and run wide, setting up a battle for the back straight."
        },
        "evolution": {
            "title": "Rain Risk",
            "desc": "The track drains poorly in spots, creating aquaplaning rivers. In dry conditions, it is hard on the right-front tyre."
        },
        "drs": {
            "title": "Uphill Drag",
            "desc": "The run to the finish line is a long, curving uphill drag race. Engine power and exit traction are vital."
        }
    },

    # --- ROUND 22: LAS VEGAS ---
    "Las Vegas Grand Prix": {
        "length_speed": {
            "title": "The Strip",
            "desc": "A cold, high-speed street track. The run down the Strip is flat out for nearly 2km. Tyre temperature is the main enemy—keeping them warm is hard."
        },
        "corners": {
            "title": "The Sphere",
            "desc": "Technical section around the Sphere requires precision, but the lap is defined by the massive straights."
        },
        "evolution": {
            "title": "Ice Skating",
            "desc": "Low temperatures and smooth asphalt make the surface feel like ice. Graining is a huge issue if the tyres slide."
        },
        "drs": {
            "title": "Slipstream King",
            "desc": "The slipstream effect on the Strip is potent. We see constant lead changes as cars tow past each other."
        }
    },

    # --- ROUND 23: QATAR ---
    "Qatar Grand Prix": {
        "length_speed": {
            "title": "MotoGP Flow",
            "desc": "Designed for bikes, it is fast and flowing. No heavy braking zones, just constant high-speed cornering load."
        },
        "corners": {
            "title": "Physicality",
            "desc": "The continuous G-force makes it the most physically demanding race for drivers. Tyres face the highest sustained loads of the year."
        },
        "evolution": {
            "title": "Kerb Danger",
            "desc": "The 'pyramid' kerbs have caused tyre failures in the past. Drivers must be careful not to run too wide."
        },
        "drs": {
            "title": "Main Straight",
            "desc": "The only real overtaking spot is the main straight. The rest of the lap is too fast to follow closely."
        }
    },

    # --- ROUND 24: ABU DHABI ---
    "Abu Dhabi Grand Prix": {
        "length_speed": {
            "title": "The Finale",
            "desc": "A Twilight race that transitions from hot sun to cool night. The modified layout is faster and more flowing than the original."
        },
        "corners": {
            "title": "The Hotel",
            "desc": "Passing under the W Hotel requires precision. The tyres are often overheating by this point in the lap."
        },
        "evolution": {
            "title": "Falling Temps",
            "desc": "As the sun sets, track temp drops, often changing the balance of the car mid-race. Rear grip improves as the night goes on."
        },
        "drs": {
            "title": "Double Zone",
            "desc": "Two long straights with DRS back-to-back allow a driver to pass and then defend (or be re-passed) immediately."
        }
    },

    # --- DEFAULT FALLBACK ---
    "default": {
        "length_speed": {"title": "Standard Circuit", "desc": "A balanced track requiring a mix of speed and cornering."},
        "corners": {"title": "Technical Mix", "desc": "Features a variety of low and high speed challenges."},
        "evolution": {"title": "Normal Evolution", "desc": "Track grip improves as rubber is laid down."},
        "drs": {"title": "Overtaking Zones", "desc": "Standard DRS zones located on main straights."}
    }
}

def get_track_detail(track_name, key):
    """Safely retrieves track data or falls back to default"""
    # Normalize keys if needed (e.g. "Bahrain" -> "Bahrain Grand Prix")
    if track_name not in TRACK_INFO:
        # Try appending " Grand Prix" if missing
        if f"{track_name} Grand Prix" in TRACK_INFO:
            track_name = f"{track_name} Grand Prix"
        else:
            return TRACK_INFO['default'][key]
            
    data = TRACK_INFO.get(track_name, TRACK_INFO['default'])
    return data.get(key, TRACK_INFO['default'][key])