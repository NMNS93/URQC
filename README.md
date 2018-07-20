# Ultrasound Reverb Quality Control (URQC)
Calculate average grayscale pixel intensity values from a curved linear ultrasound image of in-air reverberation 
patterns.

**Usage**: python3 UltrasoundReverbQC.py input(s)

<br>

**Output files**:
- plots/prefix_urqc_img.png; The image produced by contouring and used for analysis
- plots/prefix_urqc_avgs.png; A chart of the average intensity
- data/prefix_urqc_avgs.csv; The data used to produce the average intensity
- data/prefix_urqc_data.csv; The raw intensities read across the masked image

<br>

*Author(s) : Nana Mensah <Nana.mensah1@nhs.net>*

*Created : 11 April 2018*
