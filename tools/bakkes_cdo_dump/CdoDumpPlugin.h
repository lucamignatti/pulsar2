#pragma once

#include "bakkesmod/plugin/bakkesmodplugin.h"

class CdoDumpPlugin : public BakkesMod::Plugin::BakkesModPlugin
{
public:
	void onLoad() override;
	void onUnload() override;

private:
	void DumpNow();
};
