CAA_LABELS = [
    {"id": "C1", "side": "pro_govt", "name": "Glorification of the central government",
     "description": "Articles presenting the government as protector of India's sovereignty or integrity. Frames CAA/NRC as defense against illegal immigration. Appeals to national unity or security. Example: The government stands firm to protect India's borders and identity."},
    {"id": "C2", "side": "pro_govt", "name": "Vilification of the opposition",
     "description": "Articles that morally discredit or delegitimize opposition parties, portraying them as politically motivated, anti-national, or obstructive to government initiatives. Example: Opposition leaders are opposing CAA only to appease vote banks."},
    {"id": "C3", "side": "pro_govt", "name": "Glorification of CAA",
     "description": "Articles portraying the Citizenship Amendment Act itself as patriotic, humanitarian, or constitutionally sound. Emphasizes moral duty or compassion. Links CAA to India's civilizational ethos. Example: CAA offers refuge to persecuted minorities, it is an act of compassion."},
    {"id": "C4", "side": "pro_govt", "name": "Delegitimization of Critics",
     "description": "Articles suggesting critics of CAA are biased or communal. Attacks left-wing commentators or state-aligned voices. Calls out intolerance disguised as patriotism. Example: Those protesters against CAA expose their deep-rooted prejudice."},
    {"id": "C5", "side": "pro_govt", "name": "Framing anti-CAA protests as Subversive, Anti-Hindu, or Misguided",
     "description": "Articles portraying anti-CAA protesters as misinformed, anti-national, or extremist. Frames protests as threats to national unity. Suggests participants are misled or ignorant. Example: Protesters have no idea what the law even says."},
    {"id": "C6", "side": "pro_govt", "name": "Opposition spreading misinformation and fear",
     "description": "Articles accusing opposition or media of spreading falsehoods, misinterpretations, or panic regarding CAA/NRC. Mentions deliberate fear-mongering or fake news campaigns. Example: False propaganda is misleading citizens about CAA."},
    {"id": "C7", "side": "pro_govt", "name": "Anti-CAA protests are a pre-planned and funded conspiracy",
     "description": "Articles alleging anti-CAA protests are orchestrated or foreign-funded to defame India. Mentions NGOs, conspiracies, or hidden motives. Example: The protests were planned months before the Act passed. Foreign agencies are behind the chaos."},
    {"id": "C8", "side": "pro_opp", "name": "Vilification of the central government",
     "description": "Articles accusing the central government of authoritarianism, repression, or communal politics. Focuses on police brutality or suppression of dissent. Example: The government treats peaceful protesters as enemies."},
    {"id": "C9", "side": "pro_opp", "name": "Vilification of CAA",
     "description": "Articles framing the CAA law itself as discriminatory, unconstitutional, or anti-minority. Highlights exclusion of Muslims or erosion of secularism. Example: CAA legitimizes religious discrimination under the guise of compassion."},
    {"id": "C10", "side": "pro_opp", "name": "Glorifying anti-CAA protesters",
     "description": "Articles portraying anti-CAA protesters as brave defenders of democracy and justice. Describes protests as peaceful or youth-led. Example: Women of Shaheen Bagh became the conscience of the nation."},
    {"id": "C11", "side": "pro_opp", "name": "Framing anti-CAA protesters as victims",
     "description": "Articles emphasizing state violence or persecution against anti-CAA protesters. Focuses on casualties, injuries, or arrests. Example: Protesters face jail for exercising free speech."},
]

FARMERS_LABELS = [
    {"id": "F1", "side": "pro_govt", "name": "Glorification of the central government",
     "description": "Articles portraying the central government as visionary, reformist, or acting in farmers' best interest. Credits leadership for modernization. Example: The PM's reform vision will finally free farmers from middlemen."},
    {"id": "F2", "side": "pro_govt", "name": "Vilification of the opposition",
     "description": "Articles blaming opposition parties or leaders for politicizing or misleading farmers. Describes opposition leaders as hypocritical or self-serving. Example: Opposition leaders are inciting unrest for political mileage."},
    {"id": "F3", "side": "pro_govt", "name": "Justifying farm laws by critiquing current policies",
     "description": "Articles defending new farm laws by emphasizing flaws of the OLDER system, not praising the government. Frames APMC or old market structures as exploitative or monopolistic. Focus is on what was wrong BEFORE the laws. Example: For decades, farmers were trapped in the APMC monopoly. The old system benefited brokers, not farmers."},
    {"id": "F4", "side": "pro_govt", "name": "Criticizing global figures and celebrities",
     "description": "Articles discrediting international figures or celebrities who expressed solidarity with Indian farmers. Dismisses foreign commentary as ignorant or biased. Example: Foreign celebrities should stop lecturing India without understanding facts."},
    {"id": "F5", "side": "pro_opp", "name": "Vilification of the central government",
     "description": "Articles accusing the central government of arrogance, repression, or lack of empathy toward farmers. Uses moral condemnation. Example: The government silences farmers instead of hearing their pain."},
    {"id": "F6", "side": "pro_opp", "name": "Depicting farmers as victims",
     "description": "Articles portraying farmers as suffering under unjust laws or state apathy. Focuses on hardships, deaths, or sacrifices of protesters. Example: Elderly farmers spend nights on the streets for their rights."},
    {"id": "F7", "side": "pro_opp", "name": "Framing anti-farm law protests as subversive",
     "description": "Pro-opposition articles countering claims that protests are subversive, or describing state attempts to label protesters as anti-national or Khalistani. Example: The government labels peaceful farmers as Khalistanis to discredit them."},
    {"id": "F8", "side": "pro_opp", "name": "Accusing media and government of manipulation",
     "description": "Articles claiming media and government misrepresent or suppress protest realities. Mentions biased reporting or censorship. Example: The media is running government-fed lies about farmers' violence."},
    {"id": "F9", "side": "pro_opp", "name": "Emphasizing global and celebrity endorsements",
     "description": "Articles highlighting international attention or celebrity support for Indian farmers. Cites solidarity from global icons like Rihanna or Greta Thunberg. Example: Global icons from across continents have supported India's farmers."},
]

def get_labels(event):
    return CAA_LABELS if event.lower() == "caa" else FARMERS_LABELS

def get_hierarchy(event):
    return {l["id"]: l["side"] for l in get_labels(event)}
