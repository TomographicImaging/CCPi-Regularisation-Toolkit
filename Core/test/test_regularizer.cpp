unsigned char *rawData = (unsigned char *)malloc(sizeof(unsigned char) * sampleSize * nVertices);

long nSamplesRead = fread(rawData, sizeof(unsigned char), nVertices, inFile);		//	read in the block of data