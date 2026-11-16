# Sources for the disease knowledge file

The user's source list (15 Nov 2026), in the user's own order and grouping, with the status of each link
after reading on 15 Nov 2026:

- **used**: facts taken into `diseases.yaml`
- **read, no numbers**: page read, but it gave no figures for the fields we use
- **blocked**: the site refused automated reading (403, login, bot check, or an unreadable PDF)
- **not read**: skipped to meet the demo deadline

Only *used* sources back a factor. Wikipedia is used only where no other readable source on the list covered the fact: countries outside the US for gray leaf spot and northern leaf blight, and the temperature range for southern leaf blight (the USDA paper gives none). The sources are mostly about the United States, so region levels exist
mainly for US states and a few countries; everything else is "unknown" (neutral, ×1.0).

Diseases with **no temperature range** in `diseases.yaml`, because no readable source gave one: northern
leaf spot (Cornell says only "warm weather") and Curvularia leaf spot (Kentucky says "warm, humid"). Their
weather factor is neutral. Phaeosphaeria leaf spot's range comes from notes the user provided (below).

## General
- https://www.vegetables.cornell.edu/pest-management/disease-factsheets/sweet-corn-diseases-and-control-measures/ (not read)
- https://uaex.uada.edu/farm-ranch/pest-management/plant-disease/corn.aspx (not read)
- https://cropwatch.unl.edu/2018/differentiating-corn-leaf-diseases/ (used: telling gray leaf spot, NCLB, Physoderma and Goss's wilt apart)
- https://www.ars.usda.gov/research/project/?accnNo=446687 (not read)
- https://plantvillage.psu.edu/topics/corn-maize/infos (not read)
- https://www.sciencedirect.com/science/article/pii/S1537511025000984 (not read)
- https://www.britannica.com/science/plant-disease/Soil-pH (not read)

## Unrelated corn diseases (not labels in this project)
- https://cropprotectionnetwork.org/encyclopedia/diplodia-leaf-streak-of-corn (not read)
- https://farmdoc.illinois.edu/field-crop-production/diseases/another-corn-disease-alert-new-bacterial-leaf-disease-bacterial-stripe-burkholderia-andropogonis-of-corn-identified-in-illinois.html (not read; listed twice)
- https://www.cabidigitallibrary.org/doi/abs/10.1079/cabicompendium.44927 (not read)

## Healthy
- https://extension.unh.edu/resource/growing-sweet-corn-fact-sheet (not read)
- https://agcrops.osu.edu/specialization-areas/corn/how-climate-affects-corn-production (used: temperature and water needs)
- https://ucanr.edu/site/mg-sonoma/corn (not read)
- https://www.extension.purdue.edu/extmedia/nch/nch-40.html (not read)
- https://agriculture.gov.tt/wp-content/uploads/2017/11/How_to_Grow_Field_Corn-.pdf (not read)

## Rust
- https://en.wikipedia.org/wiki/Puccinia_sorghi (not read)
- https://lgpress.clemson.edu/journal/common-and-southern-rust-in-field-corn/ (used: common rust 16-25°C, April-June; southern rust 25-27°C, July-September in South Carolina; telling them apart)

### Common rust
- https://cropprotectionnetwork.org/encyclopedia/common-rust-of-corn (used: symptoms, night temperatures 65-70°F, first pustules late June)
- https://extension.umn.edu/agriculture/crop-production/corn/common-rust-on-corn (not read)
- https://www.vegetables.cornell.edu/pest-management/disease-factsheets/common-rust-of-sweet-corn/ (not read)

### Southern rust
- https://cropprotectionnetwork.org/maps/southern-corn-rust (used: timing by crop stage, tasseling onward)
- https://content.ces.ncsu.edu/corn-rusts-common-and-southern-rust (used: common rust 16-25°C + 6 h dew, throughout US and southern Canada; southern rust 25-31°C + 4 h dew, primarily southern US)
- https://cropwatch.unl.edu/plant-disease/corn/southern-rust/ (used: 80-90°F, high humidity, arrives in Nebraska early to mid July, common in the southeastern US)

## Blights
- https://www.ars.usda.gov/ARSUserFiles/60663500/Publications/Bruns/2017/Bruns_2017_Corn%20Leaf%20Blight.pdf (used: southern leaf blight is found in all corn areas, most prevalent in warm temperate and humid subtropical regions; 1970-71 epidemic. No temperature numbers)
- https://cropwatch.unl.edu/plant-disease/corn/goss-wilt/ (not read)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6640275/ (not read)
- https://apsjournals.apsnet.org/doi/10.1094/PDIS-06-21-1151-RE (not read)

### Northern corn leaf blight
- https://en.wikipedia.org/wiki/Northern_corn_leaf_blight (used for Canada, China and India and look-alikes only; temperatures agree with Purdue, Minnesota and Bayer)
- https://www.extension.purdue.edu/extmedia/bp/bp-84-w.pdf (used: 64-81°F, 6-18 h leaf wetness, late season in Indiana)
- https://cropprotectionnetwork.org/encyclopedia/northern-corn-leaf-blight-of-corn (used: 64-81°F, more than 6 h leaf wetness)
- https://www.sciencedirect.com/science/article/abs/pii/S026121941930328X (not read)
- https://extension.umn.edu/agriculture/crop-production/corn/northern-corn-leaf-blight (used: 64-80°F with prolonged moisture, at or after silking, common in the Midwestern US)
- https://www.corteva.com/us/Resources/crop-protection/disease-mgmt/managing-threat-northern-corn-leaf-blight.html (not read)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12211160/ (not read)
- https://www.labmanager.com/northern-corn-leaf-blight-genes-identified-in-new-study-5414 (not read)
- https://www.sciencedirect.com/science/article/pii/S2667064X24002781 (not read)
- https://apsjournals.apsnet.org/doi/10.1094/PHYTO-07-21-0298-R (not read)
- https://extension.sdstate.edu/northern-corn-leaf-blight-developing-corn (not read)
- https://www.cropscience.bayer.us/articles/channel/northern-corn-leaf-blight-identification-environment-and-management (used: confirms 64-81°F with moisture and high humidity)

### Southern corn leaf blight
- https://en.wikipedia.org/wiki/Southern_corn_leaf_blight (used: 68-90°F with high humidity, worldwide, 1970 US epidemic)
- https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/bipolaris-maydis (not read)
- https://cropprotectionnetwork.org/encyclopedia/southern-corn-leaf-blight-of-corn (used: pathogen, races O and T, symptoms)
- https://ui.adsabs.harvard.edu/abs/2017AgrJ..109.1218B/abstract (blocked: error 405; the same paper was read from the USDA PDF)
- (the USDA Bruns 2017 PDF is listed here again; see Blights)

## Spots
- https://corn.ces.ncsu.edu/news/dichotomous-key-for-identifying-corn-leaf-spot-disorders/ (read, no numbers: eyespot and NCLB symptoms only)
- https://apsjournals.apsnet.org/doi/10.1094/PDIS-09-21-1948-RE (not read)
- https://apsjournals.apsnet.org/doi/10.1094/PDIS-04-26-0699-PDN (not read)
- https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1873194/full (not read)
- https://link.springer.com/article/10.1007/s41348-024-01019-4 (not read)

### Gray leaf spot (this group also holds the Curvularia leaf spot sources)
- https://en.wikipedia.org/wiki/Corn_grey_leaf_spot (used for the countries outside the US only: Mexico, China, India, Philippines; its 24-35°C range was replaced by Purdue's 75-85°F)
- https://www.extension.purdue.edu/extmedia/BP/BP-56-W.pdf (used: infection at 75-85°F with more than 90% relative humidity; one of the most important leaf diseases in Indiana, worst in southern Indiana)
- https://www.farms.com/field-guide/crop-diseases/gray-leaf-spot.aspx (blocked: the reading tool refused the page)
- https://www.farmprogress.com/corn/corn-leaf-blight-can-reduce-yields-but-is-a-manageable-disease (not read)
- https://www.researchgate.net/publication/415574720_Curvularia_lunata_as_a_versatile_pathogen_causing_leaf_spot_diseases_in_diverse_plant_species_and_its_management_strategies (not read; ResearchGate usually blocks automated reading)
- https://data.mendeley.com/datasets/hmkd6nbngr/1 (not read; a dataset)
- https://www.mdpi.com/2073-431x/15/2/99 (blocked: 403)
- https://www.apsnet.org/publications/PlantDisease/BackIssues/Documents/1994Articles/PlantDisease78n10_945.PDF (blocked: 403)
- https://link.springer.com/article/10.1007/s44279-025-00275-8 (blocked: login)
- https://ieee-dataport.org/documents/annotated-maize-leaf-disease-image-dataset-cnn-based-classification (not read; a dataset)
- https://publications.mgcafe.uky.edu/sites/publications.ca.uky.edu/files/PPA-51-Curvularia-Leaf-Spot-of-Corn--Remediated.pdf (used: Curvularia leaf spot first US reports Louisiana 2017 and Kentucky 2018, important in China, warm humid, any growth stage; eyespot ~70°F, common in Michigan, Minnesota, Wisconsin, very rare in Kentucky)

### Northern leaf spot
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4363367/ (blocked: browser check)
- https://cals.cornell.edu/field-crops/corn/diseases-of-corn/northern-corn-leaf-spot (used: pathogen, symptoms at silking or maturity, warm humid weather, no numbers)
- https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/northern-corn-leaf-blight/ (not read; about northern leaf blight)
- https://rootcausereports.ces.ncsu.edu/library/0e930402-0f74-4e40-baa6-b4f3a7a8f020/ (blocked: the reading tool refused the page)
- (the Purdue BP-84 PDF is listed here again; it covers northern leaf blight; see above)
- https://www.udel.edu/content/dam/udelImages/canr/pdfs/extension/factsheets/Northern-Corn-Leaf-Blight.pdf (not read; about northern leaf blight)

## Rare spot diseases

### Eyespot
- http://extension.cropsciences.illinois.edu/fieldcrops/diseases/eye_spot/ (not read)
- https://extension.umn.edu/agriculture/crop-production/corn/eyespot-on-corn (used: pathogen, symptoms, cool wet weather, from V3/4, northern Midwest)
- https://cals.cornell.edu/field-crops/corn/diseases-of-corn/eyespot (used: cool wet weather, found across New York)
- https://webarchive.library.unt.edu/eot2008/20090116011740/http:/ohioline.osu.edu/ac-fact/pdf/0021.pdf (not read)

### Corn brown spot (Physoderma)
- https://www.researchgate.net/figure/Data-set-of-maize-foliar-diseases-A-corn-brown-spot-B-maize-curvularia-leaf-spot_fig1_370622858 (not read)
- https://cropwatch.unl.edu/plant-disease/corn/physoderma-brown-spot/ (used: symptoms, water in the whorl, minor and infrequent in Nebraska and the Midwest)
- https://cropprotectionnetwork.org/encyclopedia/physoderma-brown-spot-of-corn (used: light, free water and 75-85°F)
- https://crops.extension.iastate.edu/cropnews/2015/07/physoderma-brown-spot-and-stalk-rot (not read)
- https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/physoderma-brown-spot/ (not read)

### Tar spot (not a label in this project)
- https://ask.ifas.ufl.edu/publication/PP391#tar-spot (not read)
- https://github.com/rothmi12/TarSpot_of_Corn (not read)
- https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2025.1581312/full (not read)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6923758/ (not read)
- https://indianafieldcroppathology.com/foliar-disease-update-in-indiana-corn-and-soybean/ (not read)

### Phaeosphaeria leaf spot
- notes/phaeosphaeria-leaf-spot.md (used: notes pasted by the user on 15 Nov 2026, kept verbatim; no original citations given. Source of the 14-20°C and humidity >60-70% conditions, flowering-to-grain-fill timing, and Brazil and Mexico as regions where it is most severe)
- https://www.michiganfarmnews.com/about/news-media/msu-seeks-producer-assistance-emerging-corn-leaf-spot-disease-investigation (used: emerging in about 16 Michigan counties in 2026, similar reports from New York and Pennsylvania; cause unresolved)
- https://www.apsnet.org/publications/plantdisease/2005/January/Pages/89_1_44.aspx (blocked: 403)
- https://repository.cimmyt.org/server/api/core/bitstreams/d14102da-8e31-414e-ab2d-932f31292265/content (blocked: bot check)
- https://apsjournals.apsnet.org/doi/10.1094/PD-89-0986 (blocked: 403)
