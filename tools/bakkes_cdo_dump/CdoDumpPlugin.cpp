#include "CdoDumpPlugin.h"

#include "bakkesmod/wrappers/Engine/EngineTAWrapper.h"
#include "bakkesmod/wrappers/Engine/WorldInfoWrapper.h"
#include "bakkesmod/wrappers/GameEvent/ServerWrapper.h"
#include "bakkesmod/wrappers/GameObject/BallWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarWrapper.h"
#include "bakkesmod/wrappers/GameObject/GoalWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/AirControlComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/BoostWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/DodgeComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/DoubleJumpComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/FlipCarComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/JumpComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/VehicleSimWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/WheelWrapper.h"
#include "bakkesmod/wrappers/ArrayWrapper.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

BAKKESMOD_PLUGIN(CdoDumpPlugin, "CDO Dump", "0.2.0", PLUGINTYPE_FREEPLAY)

namespace
{
	std::string F(float v)
	{
		std::ostringstream o;
		o << std::setprecision(9) << v;
		return o.str();
	}

	std::string B(unsigned long v)
	{
		return v ? "true" : "false";
	}

	std::string Vec(const Vector& v)
	{
		std::ostringstream o;
		o << std::setprecision(9) << "(" << v.X << ", " << v.Y << ", " << v.Z << ")";
		return o.str();
	}

	std::string Rot(const Rotator& r)
	{
		std::ostringstream o;
		o << "(pitch=" << r.Pitch << ", yaw=" << r.Yaw << ", roll=" << r.Roll << ")";
		return o.str();
	}

	struct Writer
	{
		std::ofstream file;
		CVarManagerWrapper* log = nullptr;

		void line(const std::string& s)
		{
			file << s << "\n";
		}

		void kv(const char* name, const std::string& value)
		{
			line(std::string("  ") + name + " = " + value);
		}

		void kv(const char* name, float value) { kv(name, F(value)); }
		void kv(const char* name, int value) { kv(name, std::to_string(value)); }
		void kv(const char* name, unsigned char value) { kv(name, static_cast<int>(value)); }
		void kv(const char* name, unsigned long value) { kv(name, B(value)); }
		void kv(const char* name, const Vector& value) { kv(name, Vec(value)); }
		void kv(const char* name, const Rotator& value) { kv(name, Rot(value)); }

		void section(const std::string& title)
		{
			line("");
			line("[" + title + "]");
		}
	};

	void DumpJump(Writer& w, JumpComponentWrapper jump)
	{
		w.section("Jump");
		if (jump.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("MinJumpTime", jump.GetMinJumpTime());
		w.kv("JumpImpulse", jump.GetJumpImpulse());
		w.kv("JumpForce", jump.GetJumpForce());
		w.kv("JumpForceTime", jump.GetJumpForceTime());
		w.kv("PodiumJumpForceTime", jump.GetPodiumJumpForceTime());
		w.kv("JumpImpulseSpeed", jump.GetJumpImpulseSpeed());
		w.kv("JumpAccel", jump.GetJumpAccel());
		w.kv("MaxJumpHeight", jump.GetMaxJumpHeight());
		w.kv("MaxJumpHeightTime", jump.GetMaxJumpHeightTime());
	}

	void DumpDoubleJump(Writer& w, DoubleJumpComponentWrapper dj)
	{
		w.section("DoubleJump");
		if (dj.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("ImpulseScale", dj.GetImpulseScale());
		w.line("  # JumpImpulse has no getter in the Bakkes wrapper");
	}

	void DumpDodge(Writer& w, DodgeComponentWrapper dodge)
	{
		w.section("Dodge");
		if (dodge.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("DodgeInputThreshold", dodge.GetDodgeInputThreshold());
		w.kv("SideDodgeImpulse", dodge.GetSideDodgeImpulse());
		w.kv("SideDodgeImpulseMaxSpeedScale", dodge.GetSideDodgeImpulseMaxSpeedScale());
		w.kv("ForwardDodgeImpulse", dodge.GetForwardDodgeImpulse());
		w.kv("ForwardDodgeImpulseMaxSpeedScale", dodge.GetForwardDodgeImpulseMaxSpeedScale());
		w.kv("BackwardDodgeImpulse", dodge.GetBackwardDodgeImpulse());
		w.kv("BackwardDodgeImpulseMaxSpeedScale", dodge.GetBackwardDodgeImpulseMaxSpeedScale());
		w.kv("SideDodgeTorque", dodge.GetSideDodgeTorque());
		w.kv("ForwardDodgeTorque", dodge.GetForwardDodgeTorque());
		w.kv("DodgeTorqueTime", dodge.GetDodgeTorqueTime());
		w.kv("MinDodgeTorqueTime", dodge.GetMinDodgeTorqueTime());
		w.kv("DodgeZDamping", dodge.GetDodgeZDamping());
		w.kv("DodgeZDampingDelay", dodge.GetDodgeZDampingDelay());
		w.kv("DodgeZDampingUpTime", dodge.GetDodgeZDampingUpTime());
		w.kv("DodgeImpulseScale", dodge.GetDodgeImpulseScale());
		w.kv("DodgeTorqueScale", dodge.GetDodgeTorqueScale());
	}

	void DumpBoost(Writer& w, BoostWrapper boost)
	{
		w.section("Boost");
		if (boost.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("BoostConsumptionRate", boost.GetBoostConsumptionRate());
		w.kv("MaxBoostAmount", boost.GetMaxBoostAmount());
		w.kv("StartBoostAmount", boost.GetStartBoostAmount());
		w.kv("BoostModifier", boost.GetBoostModifier());
		w.kv("BoostForce", boost.GetBoostForce());
		w.kv("MinBoostTime", boost.GetMinBoostTime());
		w.kv("RechargeRate", boost.GetRechargeRate());
		w.kv("RechargeDelay", boost.GetRechargeDelay());
		w.kv("bNoBoost", boost.GetbNoBoost());
	}

	void DumpAir(Writer& w, AirControlComponentWrapper air)
	{
		w.section("AirControl");
		if (air.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("AirTorque", air.GetAirTorque());
		w.kv("AirDamping", air.GetAirDamping());
		w.kv("ThrottleForce", air.GetThrottleForce());
		w.kv("ControlScale", air.GetControlScale());
		w.kv("AirControlSensitivity", air.GetAirControlSensitivity());
	}

	void DumpFlip(Writer& w, FlipCarComponentWrapper flip)
	{
		w.section("FlipCar");
		if (flip.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.kv("FlipCarImpulse", flip.GetFlipCarImpulse());
		w.kv("FlipCarTorque", flip.GetFlipCarTorque());
		w.kv("FlipCarTime", flip.GetFlipCarTime());
	}

	void DumpWheels(Writer& w, VehicleSimWrapper sim)
	{
		if (sim.IsNull())
		{
			return;
		}
		auto wheels = sim.GetWheels();
		const int n = wheels.Count();
		w.section("Wheels n=" + std::to_string(n));
		for (int i = 0; i < n; ++i)
		{
			WheelWrapper wheel = wheels.Get(i);
			w.line("  --- wheel " + std::to_string(i) + " ---");
			if (wheel.IsNull())
			{
				w.line("    <null>");
				continue;
			}
			w.kv("WheelIndex", wheel.GetWheelIndex());
			w.kv("SteerFactor", wheel.GetSteerFactor());
			w.kv("WheelRadius", wheel.GetWheelRadius());
			w.kv("SuspensionStiffness", wheel.GetSuspensionStiffness());
			w.kv("SuspensionDampingCompression", wheel.GetSuspensionDampingCompression());
			w.kv("SuspensionDampingRelaxation", wheel.GetSuspensionDampingRelaxation());
			w.kv("SuspensionTravel", wheel.GetSuspensionTravel());
			w.kv("SuspensionMaxRaise", wheel.GetSuspensionMaxRaise());
			w.kv("ContactForceDistance", wheel.GetContactForceDistance());
			w.kv("SpinSpeedDecayRate", wheel.GetSpinSpeedDecayRate());
			w.kv("AerialThrottleToVelocityFactor", wheel.GetAerialThrottleToVelocityFactor());
			w.kv("AerialAccelerationFactor", wheel.GetAerialAccelerationFactor());
			w.kv("BoneOffset", wheel.GetBoneOffset());
			w.kv("PresetRestPosition", wheel.GetPresetRestPosition());
			w.kv("LocalSuspensionRayStart", wheel.GetLocalSuspensionRayStart());
			w.kv("LocalRestPosition", wheel.GetLocalRestPosition());
		}
	}

	void DumpCar(Writer& w, CarWrapper car, int index)
	{
		w.line("");
		w.line("======== car " + std::to_string(index) + " ========");
		if (car.IsNull())
		{
			w.line("  <null>");
			return;
		}

		w.section("Car");
		w.kv("OwnerName", car.GetOwnerName());
		w.kv("LoadoutBody", car.GetLoadoutBody());
		w.kv("AddedBallForceMultiplier", car.GetAddedBallForceMultiplier());
		w.kv("AddedCarForceMultiplier", car.GetAddedCarForceMultiplier());
		w.kv("MaxTimeForDodge", car.GetMaxTimeForDodge());
		w.kv("MaxLinearSpeed", car.GetMaxLinearSpeed());
		w.kv("MaxAngularSpeed", car.GetMaxAngularSpeed());
		w.kv("LocalCollisionOffset", car.GetLocalCollisionOffset());
		w.kv("LocalCollisionExtent", car.GetLocalCollisionExtent());
		w.kv("PreWeldMass", car.GetPreWeldMass());
		w.kv("GravityZ", car.GetGravityZ());
		w.kv("GravityAcceleration", car.GetGravityAcceleration());
		w.kv("TerminalVelocity", car.GetTerminalVelocity());
		w.kv("CustomTimeDilation", car.GetCustomTimeDilation());

		const StickyForceData sticky = car.GetStickyForce();
		w.section("StickyForce");
		w.kv("Ground", sticky.Ground);
		w.kv("Wall", sticky.Wall);

		VehicleSimWrapper sim = car.GetVehicleSim();
		w.section("VehicleSim");
		if (sim.IsNull())
		{
			w.line("  <null>");
		}
		else
		{
			w.kv("DriveTorque", sim.GetDriveTorque());
			w.kv("BrakeTorque", sim.GetBrakeTorque());
			w.kv("StopThreshold", sim.GetStopThreshold());
			w.kv("IdleBrakeFactor", sim.GetIdleBrakeFactor());
			w.kv("OppositeBrakeFactor", sim.GetOppositeBrakeFactor());
			w.kv("SteeringSensitivity", sim.GetSteeringSensitivity());
			w.kv("bUseAckermannSteering", sim.GetbUseAckermannSteering());
		}

		DumpWheels(w, sim);
		DumpJump(w, car.GetJumpComponent());
		DumpDoubleJump(w, car.GetDoubleJumpComponent());
		DumpDodge(w, car.GetDodgeComponent());
		DumpBoost(w, car.GetBoostComponent());
		DumpAir(w, car.GetAirControlComponent());
		DumpFlip(w, car.GetFlipComponent());
	}

	void DumpBall(Writer& w, BallWrapper ball, int index)
	{
		w.line("");
		w.line("======== ball " + std::to_string(index) + " ========");
		if (ball.IsNull())
		{
			w.line("  <null>");
			return;
		}
		w.section("Ball");
		w.kv("Radius", ball.GetRadius());
		w.kv("VisualRadius", ball.GetVisualRadius());
		w.kv("MagnusCoefficient", ball.GetMagnusCoefficient());
		w.kv("GroundForce", ball.GetGroundForce());
		w.kv("ReplicatedBallScale", ball.GetReplicatedBallScale());
		w.kv("ReplicatedWorldBounceScale", ball.GetReplicatedWorldBounceScale());
		w.kv("ReplicatedBallGravityScale", ball.GetReplicatedBallGravityScale());
		w.kv("ReplicatedBallMaxLinearSpeedScale", ball.GetReplicatedBallMaxLinearSpeedScale());
		w.kv("ReplicatedAddedCarBounceScale", ball.GetReplicatedAddedCarBounceScale());
		w.kv("AdditionalCarGroundBounceScaleZ", ball.GetAdditionalCarGroundBounceScaleZ());
		w.kv("AdditionalCarGroundBounceScaleXY", ball.GetAdditionalCarGroundBounceScaleXY());
		w.kv("MaxLinearSpeed", ball.GetMaxLinearSpeed());
		w.kv("MaxAngularSpeed", ball.GetMaxAngularSpeed());
		w.kv("PreWeldMass", ball.GetPreWeldMass());
		w.kv("GravityZ", ball.GetGravityZ());
		w.kv("GravityAcceleration", ball.GetGravityAcceleration());
		w.kv("TerminalVelocity", ball.GetTerminalVelocity());
		w.kv("CustomTimeDilation", ball.GetCustomTimeDilation());
	}

	void DumpEngine(Writer& w, GameWrapper* gw, ServerWrapper& server)
	{
		w.section("EngineTA");
		EngineTAWrapper eng = gw->GetEngine();
		if (!eng)
		{
			w.line("  <null>");
		}
		else
		{
			w.kv("PhysicsFramerate", eng.GetPhysicsFramerate());
			w.kv("MaxPhysicsSubsteps", eng.GetMaxPhysicsSubsteps());
			w.kv("BulletFixedDeltaTime", eng.GetBulletFixedDeltaTime());
			w.kv("PhysicsTime", eng.GetPhysicsTime());
			w.kv("PhysicsFrame", eng.GetPhysicsFrame());
			w.kv("ReplicatedPhysicsFrame", eng.GetReplicatedPhysicsFrame());
			w.kv("LastPhysicsDeltaTimeScale", eng.GetLastPhysicsDeltaTimeScale());
			w.kv("bEnableClientPrediction", eng.GetbEnableClientPrediction());
			w.kv("bClientPhysicsUpdate", eng.GetbClientPhysicsUpdate());
			w.kv("bDisableClientCorrections", eng.GetbDisableClientCorrections());
			w.kv("MaxUploadedClientFrames", eng.GetMaxUploadedClientFrames());
			w.kv("MaxClientReplayFrames", eng.GetMaxClientReplayFrames());
		}

		w.section("WorldInfo");
		WorldInfoWrapper world = server.GetWorldInfo();
		if (world.IsNull())
		{
			w.line("  <null>");
		}
		else
		{
			w.kv("WorldGravityZ", world.GetWorldGravityZ());
			w.kv("DefaultGravityZ", world.GetDefaultGravityZ());
			w.kv("GlobalGravityZ", world.GetGlobalGravityZ());
			w.kv("RBPhysicsGravityScaling", world.GetRBPhysicsGravityScaling());
			w.kv("StallZ", world.GetStallZ());
			w.kv("TimeDilation", world.GetTimeDilation());
			w.kv("DeltaSeconds", world.GetDeltaSeconds());
			w.kv("RealDeltaSeconds", world.GetRealDeltaSeconds());
		}

		w.section("Server");
		w.kv("GameSpeed", server.GetGameSpeed());
		w.kv("MatchTimeDilation", server.GetMatchTimeDilation());
	}

	void DumpGoals(Writer& w, ServerWrapper& server)
	{
		auto goals = server.GetGoals();
		const int n = goals.Count();
		w.line("");
		w.line("# goals = " + std::to_string(n));
		for (int i = 0; i < n; ++i)
		{
			GoalWrapper goal = goals.Get(i);
			w.line("");
			w.line("======== goal " + std::to_string(i) + " ========");
			if (!goal)
			{
				w.line("  <null>");
				continue;
			}
			w.section("Goal");
			w.kv("TeamNum", goal.GetTeamNum());
			w.kv("PointsToAward", goal.GetPointsToAward());
			w.kv("Location", goal.GetLocation());
			w.kv("Direction", goal.GetDirection());
			w.kv("Up", goal.GetUp());
			w.kv("LocalExtent", goal.GetLocalExtent());
			w.kv("WorldCenter", goal.GetWorldCenter());
			w.kv("WorldExtent", goal.GetWorldExtent());
			w.kv("WorldFrontCenter", goal.GetWorldFrontCenter());
		}
	}
}

void CdoDumpPlugin::onLoad()
{
	cvarManager->registerNotifier(
		"rs_dump_cdo",
		[this](std::vector<std::string>) {
			gameWrapper->Execute([this](GameWrapper*) { DumpNow(); });
		},
		"Dump live Bakkes physics wrapper fields (CDO-shaped numbers) to data/cdo_dump/",
		PERMISSION_ALL);
	cvarManager->log("CdoDumpPlugin loaded. In a match: rs_dump_cdo");
}

void CdoDumpPlugin::onUnload() {}

void CdoDumpPlugin::DumpNow()
{
	if (gameWrapper->IsInOnlineGame())
	{
		cvarManager->log("rs_dump_cdo: refuse online game. Use freeplay / private / training.");
		return;
	}

	ServerWrapper server = gameWrapper->GetGameEventAsServer();
	if (server.IsNull())
	{
		server = gameWrapper->GetCurrentGameState();
	}
	if (server.IsNull())
	{
		cvarManager->log("rs_dump_cdo: no server (enter freeplay or a private match first).");
		return;
	}

	const auto now = std::chrono::system_clock::now();
	const std::time_t t = std::chrono::system_clock::to_time_t(now);
	std::tm local{};
	localtime_s(&local, &t);
	char stamp[32];
	std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);

	std::filesystem::path dir = gameWrapper->GetDataFolder() / "cdo_dump";
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	const std::filesystem::path path = dir / (std::string("cdo_") + stamp + ".txt");

	Writer w;
	w.log = cvarManager.get();
	w.file.open(path);
	if (!w.file)
	{
		cvarManager->log("rs_dump_cdo: failed to write " + path.string());
		return;
	}

	w.line("# CdoDumpPlugin 0.2.0");
	w.line("# Live instance fields via Bakkes wrappers, not a UProperty walk.");
	w.line("# Constants should match the class default object unless the instance mutated them.");
	w.line(std::string("# psy_build = ") + gameWrapper->GetPsyBuildID());
	w.line(std::string("# map = ") + gameWrapper->GetCurrentMap());
	w.line(std::string("# epic = ") + (gameWrapper->IsUsingEpicVersion() ? "true" : "false"));
	w.line(std::string("# freeplay = ") + (gameWrapper->IsInFreeplay() ? "true" : "false"));
	w.line(std::string("# custom_training = ") + (gameWrapper->IsInCustomTraining() ? "true" : "false"));

	DumpEngine(w, gameWrapper.get(), server);

	auto cars = server.GetCars();
	w.line(std::string("# cars = ") + std::to_string(cars.Count()));
	for (int i = 0; i < cars.Count(); ++i)
	{
		DumpCar(w, cars.Get(i), i);
	}

	auto balls = server.GetGameBalls();
	w.line(std::string("# balls = ") + std::to_string(balls.Count()));
	for (int i = 0; i < balls.Count(); ++i)
	{
		DumpBall(w, balls.Get(i), i);
	}

	DumpGoals(w, server);

	w.file.close();
	cvarManager->log("rs_dump_cdo wrote " + path.string());
}
