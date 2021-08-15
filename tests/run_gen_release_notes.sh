#!/bin/bash
docker run -it --rm -v "$(pwd)":/usr/local/src/your-app/ githubchangeloggenerator/github-changelog-generator \
	-u nicolay-r \
	-p AREkit \
	--token <GITHUB_TOKEN> \
	--since-tag v0.20.5-rc
