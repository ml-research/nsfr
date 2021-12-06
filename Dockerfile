# Select the base image
FROM nvcr.io/nvidia/pytorch:21.06-py3

# Select the working directory
WORKDIR  /NSFR

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add fonts for serif rendering in MPL plots
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install --yes ttf-mscorefonts-installer
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN apt-get install dvipng cm-super fonts-cmu --yes
RUN apt-get install fonts-dejavu-core --yes
