#!/usr/bin/env python3

from typing import List, Dict

# QA PROMPTS - Question Answering
# Domande che richiedono risposte informative e dettagliate.
# Il modello deve fornire spiegazioni, definizioni, o analisi.

QA_PROMPTS: List[str] = [
    # --- Scienza e Tecnologia ---
    "What is machine learning and how does it differ from traditional programming?",
    "Explain the concept of neural networks in simple terms.",
    "How does photosynthesis work in plants?",
    "What is quantum computing and why is it important?",
    "Describe the process of DNA replication.",
    "What causes climate change and what are its main effects?",
    "How do vaccines work to protect against diseases?",
    "Explain the theory of relativity in simple terms.",
    "What is artificial intelligence and what are its main applications?",
    "How does the internet work at a basic level?",
    
    # --- Storia e Società ---
    "What were the main causes of World War I?",
    "How did the Industrial Revolution change society?",
    "What was the significance of the Renaissance period?",
    "Explain the concept of democracy and its origins.",
    "What factors led to the fall of the Roman Empire?",
    "How did the printing press change the world?",
    "What were the main achievements of ancient Egyptian civilization?",
    "Describe the impact of the French Revolution on modern politics.",
    "What was the Cold War and why was it significant?",
    "How did ancient Greek philosophy influence Western thought?",
    
    # --- Economia e Business ---
    "What is inflation and how does it affect the economy?",
    "Explain the concept of supply and demand.",
    "How do stock markets work?",
    "What is the difference between GDP and GNP?",
    "Explain the role of central banks in the economy.",
    "What causes economic recessions?",
    "How does international trade benefit countries?",
    "What is cryptocurrency and how does it work?",
    "Explain the concept of compound interest.",
    "What are the main functions of money in an economy?",
    
    # --- Natura e Ambiente ---
    "How do ecosystems maintain balance?",
    "What causes earthquakes and how are they measured?",
    "Explain the water cycle and its importance.",
    "How do animals adapt to their environments?",
    "What is biodiversity and why is it important?",
    "How do volcanoes form and erupt?",
    "Explain the greenhouse effect and its consequences.",
    "What role do bees play in the ecosystem?",
    "How do coral reefs support marine life?",
    "What causes seasons to change throughout the year?",
    
    # --- Salute e Medicina ---
    "How does the human immune system work?",
    "What is the difference between bacteria and viruses?",
    "Explain how antibiotics work against infections.",
    "What causes heart disease and how can it be prevented?",
    "How does the brain process and store memories?",
    "What is the role of sleep in human health?",
    "Explain how genetic inheritance works.",
    "What causes allergies and how do they develop?",
    "How does exercise benefit mental health?",
    "What is the placebo effect and why does it occur?",
]

# SUMMARY PROMPTS - Summarization
# Testi da riassumere. Il modello deve produrre un riassunto conciso.
# Formato: "Summarize the following text: [testo]"

SUMMARY_PROMPTS: List[str] = [
    # --- Articoli Scientifici ---
    "Summarize the following text: The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine. Fleming noticed that a mold called Penicillium notatum had contaminated one of his petri dishes and was killing the bacteria around it. This accidental discovery led to the development of antibiotics, which have saved countless lives by treating bacterial infections that were once fatal. The mass production of penicillin during World War II marked the beginning of the antibiotic era.",
    
    "Summarize the following text: Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors have caused climate variations throughout Earth's history, human activities have been the main driver since the Industrial Revolution. The burning of fossil fuels releases greenhouse gases that trap heat in the atmosphere, leading to global warming. Effects include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems worldwide.",
    
    "Summarize the following text: The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This complex network enables all human thought, emotion, and behavior. Different regions of the brain specialize in different functions: the frontal lobe handles reasoning and planning, the temporal lobe processes language and memory, and the cerebellum coordinates movement. Understanding how these regions work together remains one of science's greatest challenges.",
    
    "Summarize the following text: Artificial intelligence has made remarkable progress in recent years, particularly in the field of deep learning. Neural networks can now recognize images, understand speech, and generate human-like text with impressive accuracy. These advances have enabled applications from self-driving cars to medical diagnosis. However, concerns about AI safety, bias, and job displacement have prompted calls for careful regulation and ethical guidelines.",
    
    "Summarize the following text: The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over time. Organisms with traits better suited to their environment are more likely to survive and reproduce, passing these advantageous traits to their offspring. Over many generations, this process can lead to the emergence of new species. Fossil evidence and genetic studies have provided overwhelming support for evolutionary theory.",
    
    # --- Storia e Politica ---
    "Summarize the following text: The Renaissance was a cultural movement that began in Italy during the 14th century and spread throughout Europe. It marked a transition from medieval to modern times, characterized by renewed interest in classical Greek and Roman culture. Artists like Leonardo da Vinci and Michelangelo created masterpieces that still inspire today. The period also saw advances in science, literature, and philosophy that laid the groundwork for the modern world.",
    
    "Summarize the following text: The United Nations was founded in 1945 after World War II to promote international cooperation and prevent future conflicts. Its main organs include the General Assembly, Security Council, and International Court of Justice. The UN addresses issues ranging from peacekeeping and humanitarian aid to human rights and sustainable development. While criticized for limitations in enforcement, it remains the world's primary forum for international diplomacy.",
    
    "Summarize the following text: The Industrial Revolution transformed society from agrarian economies to industrial powerhouses. Beginning in Britain in the late 18th century, it introduced factory production, steam power, and mechanized manufacturing. While it brought unprecedented economic growth and technological innovation, it also created harsh working conditions, urban overcrowding, and environmental pollution. The social changes it triggered continue to shape our world today.",
    
    "Summarize the following text: Democracy originated in ancient Athens around the 5th century BCE, where citizens could directly participate in government decisions. Modern democracies typically use representative systems where elected officials make decisions on behalf of constituents. Key principles include free and fair elections, protection of individual rights, separation of powers, and rule of law. Democracy has spread globally but faces ongoing challenges from authoritarianism and populism.",
    
    "Summarize the following text: The space race between the United States and Soviet Union during the Cold War drove remarkable achievements in space exploration. The Soviets launched the first satellite, Sputnik, in 1957 and sent the first human, Yuri Gagarin, into space in 1961. The Americans responded with the Apollo program, landing astronauts on the Moon in 1969. This competition accelerated technological development and inspired generations of scientists and engineers.",
    
    # --- Tecnologia ---
    "Summarize the following text: The internet began as a military research project called ARPANET in the 1960s. It evolved into a global network connecting billions of devices through standardized protocols. The World Wide Web, invented by Tim Berners-Lee in 1989, made the internet accessible to ordinary users through web browsers. Today, the internet underpins modern communication, commerce, entertainment, and information sharing, fundamentally changing how society functions.",
    
    "Summarize the following text: Blockchain technology provides a decentralized, tamper-resistant way to record transactions. Each block contains a cryptographic hash of the previous block, creating an unbreakable chain. Originally developed for Bitcoin, blockchain now has applications in supply chain management, voting systems, and digital identity verification. Its potential to eliminate intermediaries and increase transparency has attracted significant investment and research.",
    
    "Summarize the following text: Electric vehicles represent a significant shift in transportation technology. Unlike conventional cars that burn gasoline, EVs use electric motors powered by rechargeable batteries. Advances in battery technology have increased range and reduced costs, making EVs more practical for consumers. Environmental benefits include zero direct emissions, though the overall impact depends on how electricity is generated. Major automakers are investing heavily in EV development.",
    
    "Summarize the following text: Cloud computing delivers computing services over the internet, including storage, processing power, and software applications. Instead of maintaining local servers, businesses can rent resources from providers like Amazon Web Services or Microsoft Azure. Benefits include scalability, cost efficiency, and accessibility from anywhere. Security concerns and dependence on internet connectivity remain challenges, but cloud adoption continues to grow rapidly.",
    
    "Summarize the following text: Social media platforms have transformed how people communicate and consume information. Sites like Facebook, Twitter, and Instagram allow users to share content and connect with others globally. While social media has democratized information sharing and enabled social movements, it has also been criticized for spreading misinformation, enabling harassment, and negatively affecting mental health, particularly among young users.",
    
    # --- Economia ---
    "Summarize the following text: Globalization refers to the increasing interconnection of economies, cultures, and populations worldwide. International trade, investment, and migration have accelerated this process. Supporters argue globalization promotes economic growth and cultural exchange. Critics point to job losses in developed countries, exploitation of workers in developing nations, and environmental degradation. The COVID-19 pandemic highlighted both the benefits and vulnerabilities of global supply chains.",
    
    "Summarize the following text: Inflation occurs when the general price level of goods and services rises over time, reducing purchasing power. Central banks typically aim for low, stable inflation around 2% annually. Causes include excess money supply, demand exceeding supply, and rising production costs. High inflation erodes savings and creates economic uncertainty, while deflation can lead to reduced spending and economic stagnation. Managing inflation is a key challenge for policymakers.",
    
    "Summarize the following text: The gig economy describes a labor market characterized by short-term contracts and freelance work rather than permanent employment. Platforms like Uber, Airbnb, and Upwork connect workers directly with customers. Benefits include flexibility and entrepreneurial opportunities. However, gig workers often lack benefits like health insurance and retirement plans, leading to debates about worker classification and labor protections in this new economy.",
    
    "Summarize the following text: Sustainable development aims to meet present needs without compromising future generations' ability to meet their own needs. It balances economic growth, environmental protection, and social well-being. The United Nations' Sustainable Development Goals provide a framework covering poverty, education, health, climate action, and more. Achieving sustainability requires changes in production, consumption, and governance at all levels of society.",
    
    "Summarize the following text: Cryptocurrency is digital currency that uses cryptography for security and operates on decentralized networks. Bitcoin, created in 2009, was the first cryptocurrency and remains the most valuable. Transactions are recorded on a blockchain, eliminating the need for banks or governments as intermediaries. While some see cryptocurrency as the future of finance, others worry about volatility, energy consumption, and use in illegal activities.",
    
    # --- Scienza ---
    "Summarize the following text: CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists to precisely modify DNA sequences. Derived from a bacterial immune system, it works like molecular scissors to cut DNA at specific locations. Applications include treating genetic diseases, creating disease-resistant crops, and studying gene function. Ethical concerns about editing human embryos and creating designer babies have prompted calls for careful regulation.",
    
    "Summarize the following text: Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their lives. Despite being invisible, black holes can be detected by their gravitational effects on nearby matter. The first image of a black hole was captured in 2019, confirming predictions of Einstein's general relativity and opening new avenues for astrophysical research.",
    
    "Summarize the following text: Renewable energy sources like solar, wind, and hydroelectric power offer alternatives to fossil fuels. Solar panels convert sunlight directly into electricity, while wind turbines harness kinetic energy from moving air. These technologies have become increasingly cost-competitive and are essential for reducing greenhouse gas emissions. Challenges include intermittency, storage, and the need for grid infrastructure upgrades to accommodate variable generation.",
    
    "Summarize the following text: The microbiome refers to the trillions of microorganisms living in and on the human body, particularly in the gut. These bacteria, viruses, and fungi play crucial roles in digestion, immune function, and even mental health. Research has linked microbiome imbalances to conditions including obesity, diabetes, and depression. Probiotics and dietary changes can influence microbiome composition, though optimal interventions are still being studied.",
    
    "Summarize the following text: Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales. Particles can exist in multiple states simultaneously until observed, a phenomenon called superposition. Entanglement allows particles to be correlated regardless of distance. While counterintuitive, quantum mechanics has been experimentally verified and underlies technologies from lasers to semiconductors. Quantum computers promise to solve problems beyond classical computers' capabilities.",
    
    # --- Cultura e Società ---
    "Summarize the following text: Urbanization is the increasing concentration of populations in cities. More than half of the world's population now lives in urban areas, a figure expected to reach 68% by 2050. Cities offer economic opportunities, better services, and cultural amenities. However, rapid urbanization strains infrastructure, creates housing shortages, and contributes to pollution. Sustainable urban planning is essential to manage growth while maintaining quality of life.",
    
    "Summarize the following text: Mental health awareness has increased significantly in recent years. Conditions like depression and anxiety affect millions worldwide, yet stigma often prevents people from seeking help. Treatment options include therapy, medication, and lifestyle changes. Workplace mental health programs and school-based interventions aim to promote well-being and early intervention. Advocates call for mental health to be treated with the same importance as physical health.",
    
    "Summarize the following text: Education systems worldwide face pressure to prepare students for a rapidly changing job market. Traditional models emphasizing memorization are giving way to approaches focused on critical thinking, creativity, and digital literacy. Online learning has expanded access but raised questions about quality and equity. Lifelong learning is increasingly important as technological change requires workers to continuously update their skills.",
    
    "Summarize the following text: Food security means ensuring all people have access to sufficient, safe, and nutritious food. Despite producing enough food globally, nearly 800 million people face hunger due to poverty, conflict, and distribution challenges. Climate change threatens agricultural productivity, while population growth increases demand. Solutions include sustainable farming practices, reduced food waste, and policies addressing inequality in food access.",
    
    "Summarize the following text: Privacy concerns have grown as technology enables unprecedented data collection. Companies gather information about online behavior, location, and purchases to target advertising and improve services. Governments conduct surveillance for security purposes. Regulations like GDPR aim to protect personal data and give individuals more control. Balancing privacy with the benefits of data-driven innovation remains a significant challenge for society.",
    
    # --- Più prompt per raggiungere 50 ---
    "Summarize the following text: Automation and artificial intelligence are transforming the workforce. Robots handle manufacturing tasks, algorithms make financial decisions, and chatbots provide customer service. While automation increases productivity and reduces costs, it also displaces workers in certain industries. Economists debate whether new jobs will emerge to replace those lost. Preparing workers through education and retraining programs is crucial for managing this transition.",
    
    "Summarize the following text: Biodiversity loss threatens ecosystems worldwide. Species are going extinct at rates far exceeding natural background levels, driven by habitat destruction, pollution, climate change, and overexploitation. The loss of biodiversity reduces ecosystem resilience and threatens services humans depend on, including pollination, water purification, and climate regulation. Conservation efforts focus on protecting habitats, restoring ecosystems, and sustainable resource management.",
    
    "Summarize the following text: The aging population presents challenges and opportunities for societies worldwide. Advances in healthcare have extended life expectancy, but declining birth rates mean fewer workers supporting more retirees. Healthcare and pension systems face increased costs. However, older adults contribute through volunteering, caregiving, and continued employment. Policies promoting healthy aging and intergenerational solidarity are essential for adapting to demographic change.",
    
    "Summarize the following text: Water scarcity affects billions of people worldwide and is expected to worsen with climate change and population growth. Agriculture consumes the majority of freshwater resources. Solutions include improved irrigation efficiency, desalination, water recycling, and conservation measures. International cooperation is necessary for managing shared water resources. Access to clean water is recognized as a human right, yet many lack this basic necessity.",
    
    "Summarize the following text: Plastic pollution has become a global environmental crisis. Millions of tons of plastic enter oceans annually, harming marine life and entering the food chain. Microplastics have been found in drinking water, seafood, and even human blood. Solutions include reducing single-use plastics, improving recycling systems, and developing biodegradable alternatives. International agreements aim to address plastic pollution, but implementation remains challenging.",
    
    "Summarize the following text: Nuclear energy provides about 10% of global electricity without direct carbon emissions. Modern reactors are safer than older designs, and new technologies promise further improvements. However, concerns about accidents, radioactive waste disposal, and nuclear proliferation persist. Some countries are expanding nuclear capacity to meet climate goals, while others are phasing it out. The debate over nuclear energy's role in the energy transition continues.",
    
    "Summarize the following text: Telemedicine enables healthcare delivery through digital communication technologies. Patients can consult doctors via video calls, receive remote monitoring, and access health information online. The COVID-19 pandemic accelerated telemedicine adoption. Benefits include increased access, convenience, and reduced costs. Challenges include ensuring quality of care, protecting patient privacy, and addressing the digital divide that limits access for some populations.",
    
    "Summarize the following text: Antibiotic resistance is a growing threat to global health. Overuse and misuse of antibiotics have allowed bacteria to evolve resistance, making infections harder to treat. Common infections could become deadly without effective antibiotics. Addressing this requires reducing unnecessary antibiotic use, developing new drugs, and improving infection prevention. International coordination is essential as resistant bacteria spread across borders.",
    
    "Summarize the following text: Space exploration has entered a new era with private companies joining government agencies. SpaceX, Blue Origin, and others are reducing launch costs and developing reusable rockets. Goals include returning humans to the Moon, establishing Mars colonies, and mining asteroids. Space tourism is becoming reality for wealthy individuals. These developments raise questions about space governance, resource rights, and the commercialization of space.",
    
    "Summarize the following text: Artificial intelligence in healthcare offers transformative potential. Machine learning algorithms can analyze medical images, predict patient outcomes, and accelerate drug discovery. AI assistants help with diagnosis and treatment planning. However, concerns include algorithmic bias, liability for errors, and the changing role of healthcare professionals. Ensuring AI benefits patients while maintaining safety and trust requires careful implementation and regulation.",
    
    "Summarize the following text: The circular economy aims to eliminate waste by keeping materials in use as long as possible. Unlike the traditional linear model of make-use-dispose, circular approaches emphasize repair, reuse, remanufacturing, and recycling. Benefits include reduced resource extraction, lower emissions, and new business opportunities. Implementing circular economy principles requires redesigning products, changing consumer behavior, and developing new business models.",
    
    "Summarize the following text: Disinformation and misinformation spread rapidly through social media, threatening public health, democracy, and social cohesion. False information about vaccines, elections, and other topics can have serious consequences. Platforms use fact-checking and content moderation, but balancing free speech with harm prevention is challenging. Media literacy education helps people evaluate information critically. Addressing disinformation requires cooperation among platforms, governments, and civil society.",
    
    "Summarize the following text: Gender equality has advanced significantly but remains incomplete worldwide. Women have gained rights to vote, own property, and access education in many countries. However, gaps persist in pay, leadership representation, and unpaid care work. Violence against women remains prevalent. Achieving gender equality requires addressing cultural attitudes, discriminatory laws, and structural barriers. Evidence shows gender equality benefits economic growth and social development.",
    
    "Summarize the following text: The future of work is being shaped by technology, demographics, and changing expectations. Remote work has become mainstream for many knowledge workers. Automation is changing job requirements across industries. Workers increasingly value flexibility, purpose, and work-life balance. Organizations must adapt their cultures and practices to attract and retain talent. The social contract between employers and workers is being renegotiated in this new environment.",
    
    "Summarize the following text: Ocean acidification, caused by absorption of atmospheric carbon dioxide, threatens marine ecosystems. As oceans become more acidic, organisms that build calcium carbonate shells and skeletons struggle to survive. Coral reefs, shellfish, and plankton are particularly vulnerable. Since marine ecosystems support fisheries that feed billions of people, ocean acidification has significant implications for food security. Reducing carbon emissions is essential to address this problem.",
    
    "Summarize the following text: Virtual reality technology has evolved from a niche gaming accessory to a tool with applications across industries. In healthcare, VR helps treat phobias and PTSD through exposure therapy. Architects use it to walk clients through building designs before construction. Training programs in aviation, surgery, and emergency response benefit from realistic simulations without real-world risks. As hardware becomes more affordable and content more diverse, VR adoption continues to expand.",
    
    "Summarize the following text: Sleep plays a crucial role in physical and mental health that scientists are only beginning to fully understand. During sleep, the brain consolidates memories and clears toxic waste products. Chronic sleep deprivation is linked to obesity, diabetes, cardiovascular disease, and cognitive decline. Despite this, modern lifestyles often prioritize productivity over rest. Public health experts recommend adults get seven to nine hours of sleep per night for optimal health.",
    
    "Summarize the following text: Urbanization continues to reshape human civilization, with more than half the world's population now living in cities. Urban areas offer economic opportunities, cultural amenities, and efficient delivery of services like healthcare and education. However, rapid growth strains infrastructure, creates housing shortages, and contributes to pollution and congestion. Smart city technologies using sensors and data analytics promise to improve urban management, though privacy concerns persist.",
    
    "Summarize the following text: The human microbiome consists of trillions of bacteria, viruses, and fungi living in our bodies, particularly the gut. These microorganisms help digest food, produce vitamins, and train the immune system. Research has linked microbiome imbalances to conditions ranging from obesity to depression. Diet, antibiotics, and lifestyle factors influence microbiome composition. Scientists are exploring how manipulating the microbiome through probiotics and dietary interventions could improve health outcomes.",
    
    "Summarize the following text: Artificial intelligence is transforming healthcare in numerous ways. Machine learning algorithms can analyze medical images to detect diseases earlier than human physicians. Natural language processing helps extract insights from clinical notes and research papers. AI-powered tools assist in drug discovery by predicting how molecules will interact. While promising, AI in healthcare raises concerns about algorithmic bias, liability, and the changing role of medical professionals.",
]

# NEWS PROMPTS - News Generation
# Prompt per generare articoli, notizie, comunicati stampa.
# Questi possono produrre output di lunghezza variabile.

NEWS_PROMPTS: List[str] = [
    # --- Tecnologia ---
    "Write a news article about a breakthrough in artificial intelligence research.",
    "Write a news article about a new smartphone launch by a major tech company.",
    "Write a news article about cybersecurity threats affecting businesses.",
    "Write a news article about advances in electric vehicle technology.",
    "Write a news article about a startup that raised significant funding.",
    "Write a news article about changes in social media platform policies.",
    "Write a news article about developments in quantum computing.",
    "Write a news article about the impact of automation on employment.",
    "Write a news article about a major tech company's quarterly earnings.",
    "Write a news article about privacy concerns with new technology.",
    
    # --- Scienza ---
    "Write a news article about a new medical treatment discovery.",
    "Write a news article about climate research findings.",
    "Write a news article about a space exploration milestone.",
    "Write a news article about archaeological discoveries.",
    "Write a news article about advances in renewable energy.",
    "Write a news article about a breakthrough in cancer research.",
    "Write a news article about endangered species conservation efforts.",
    "Write a news article about a volcanic eruption and its effects.",
    "Write a news article about discoveries in deep ocean exploration.",
    "Write a news article about a new study on nutrition and health.",
    
    # --- Economia ---
    "Write a news article about stock market movements.",
    "Write a news article about central bank interest rate decisions.",
    "Write a news article about a major corporate merger.",
    "Write a news article about international trade negotiations.",
    "Write a news article about unemployment statistics.",
    "Write a news article about housing market trends.",
    "Write a news article about inflation affecting consumers.",
    "Write a news article about a company announcing layoffs.",
    "Write a news article about cryptocurrency market developments.",
    "Write a news article about government budget proposals.",
    
    # --- Politica e Società ---
    "Write a news article about election results in a democratic country.",
    "Write a news article about a new government policy on education.",
    "Write a news article about international diplomatic relations.",
    "Write a news article about protests over social issues.",
    "Write a news article about immigration policy changes.",
    "Write a news article about healthcare system reforms.",
    "Write a news article about environmental regulations.",
    "Write a news article about a court ruling on civil rights.",
    "Write a news article about a summit between world leaders.",
    "Write a news article about efforts to address homelessness.",
    
    # --- Sport e Cultura ---
    "Write a news article about a major sports championship.",
    "Write a news article about a film winning prestigious awards.",
    "Write a news article about a music festival announcement.",
    "Write a news article about a famous athlete's retirement.",
    "Write a news article about a museum opening a new exhibition.",
    "Write a news article about a best-selling book release.",
    "Write a news article about an e-sports tournament.",
    "Write a news article about a theater production premiere.",
    "Write a news article about a celebrity charity initiative.",
    "Write a news article about trends in streaming entertainment.",
]

# FUNZIONI DI ACCESSO
def get_prompts_for_task(task: str) -> List[str]:
    task_lower = task.lower()
    
    if task_lower == 'qa':
        return QA_PROMPTS
    elif task_lower == 'summary':
        return SUMMARY_PROMPTS
    elif task_lower == 'news':
        return NEWS_PROMPTS
    else:
        raise ValueError(f"Task '{task}' non riconosciuto. Usa 'qa', 'summary', o 'news'.")


def get_all_prompts() -> Dict[str, List[str]]:
    return {
        'qa': QA_PROMPTS,
        'summary': SUMMARY_PROMPTS,
        'news': NEWS_PROMPTS
    }


def get_prompt_stats() -> Dict[str, int]:
    return {
        'qa': len(QA_PROMPTS),
        'summary': len(SUMMARY_PROMPTS),
        'news': len(NEWS_PROMPTS),
        'total': len(QA_PROMPTS) + len(SUMMARY_PROMPTS) + len(NEWS_PROMPTS)
    }


# TEST
if __name__ == "__main__":
    print("=" * 70)
    print("PROMPT COLLECTION - Statistiche")
    print("=" * 70)
    
    stats = get_prompt_stats()
    print(f"\nPrompt per task:")
    print(f"  QA:      {stats['qa']} prompt")
    print(f"  Summary: {stats['summary']} prompt")
    print(f"  News:    {stats['news']} prompt")
    print(f"  TOTALE:  {stats['total']} prompt")
    
    print(f"\nEsempi di prompt:")
    print(f"\n  QA (primo):")
    print(f"    '{QA_PROMPTS[0][:70]}...'")
    print(f"\n  Summary (primo):")
    print(f"    '{SUMMARY_PROMPTS[0][:70]}...'")
    print(f"\n  News (primo):")
    print(f"    '{NEWS_PROMPTS[0][:70]}...'")
    
    # Verifica lunghezza
    print(f"\nVerifica completezza:")
    for task, count in [('qa', 50), ('summary', 50), ('news', 50)]:
        prompts = get_prompts_for_task(task)
        status = "✓" if len(prompts) >= count else "✗"
        print(f"  {status} {task}: {len(prompts)}/{count}")
    
    print("\n" + "=" * 70)
