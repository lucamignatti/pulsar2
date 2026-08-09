public function bool ShouldDemolish(Car_TA HitCar, Vector HitLocation, Vector HitNormal, out enum ECarImpactResult Result) { 
    local Vector CarForward; 
    local float ForwardSpeed, DemoSpeedRequired; 
    local bool bAllowBackwardsDemolitions; 
    local TimeOfImpactData Impact; 
    local bool bReverseCarForward; 
    if (DemolishTarget == EDemolishTarget.DemolishTarget_None) { 
        Result = ECarImpactResult.CarImpactResult_FailDemolishTargetNone; 
        return false; 
    } 
    if (DemolishSpeed > EDemolishSpeed.DemolishSpeed_None) { 
        if (DemolishSpeed == EDemolishSpeed.DemolishSpeed_Supersonic && bSuperSonic == false) { 
            Result = ECarImpactResult.CarImpactResult_FailNotSupersonic; 
            return false; 
        } 
        CarForward = Vector(QuatToRotator(OldRBState.Quaternion)); 
        ForwardSpeed = OldRBState.LinearVelocity Dot CarForward; 
        DemoSpeedRequired = PhysicsConfig.SuperSonicSettings.Speed - PhysicsConfig.SuperSonicSettings.TurnoffSpeedBuffer; 
        bAllowBackwardsDemolitions = PhysicsConfig.bAllowBackwardsDemolitions != 0; 
        if (bAllowBackwardsDemolitions) { 
            ForwardSpeed = Abs(ForwardSpeed); 
        } 
        DemoSpeedRequired = PhysicsConfig.GetSpeedRequiredForDemo(DemolishSpeed); 
        if (ForwardSpeed < DemoSpeedRequired) { 
            Result = ECarImpactResult.CarImpactResult_FailInsufficientForwardSpeed; 
            return false; 
        } 
        GetTimeOfImpact(HitCar, Impact); 
        if (Impact.Fraction >= 1) { 
            LogInternal("ShouldDemolish() sweep check missed."); 
            InitTimeOfImpactFromOldRBState(HitCar, HitLocation, HitNormal, Impact); 
            goto label_1287; 
        } 
        if (bDebug) { 
            DrawCollisionBox(Impact.Location, Impact.Rotation, MakeColor(255, 0, 0), true); 
            DrawDebugLine(Impact.Location + LocalCollisionOffset >> Impact.Rotation, Impact.ImpactLocation, 255, 0, 0, true); 
            HitCar.DrawCollisionBox(Impact.OtherLocation, Impact.OtherRotation, MakeColor(0, 255, 0), true); 
            DrawDebugLine(Impact.OtherLocation + HitCar.LocalCollisionOffset >> Impact.OtherRotation, Impact.ImpactLocation, 0, 255, 0, true); 
            DrawDebugBox(Impact.ImpactLocation, MakeVector(1, 1, 1), 0, 255, 255, true); 
            DrawDebugLine(Impact.ImpactLocation, Impact.ImpactLocation + Impact.ImpactNormal * 10, 0, 255, 255, true); 
        } 
label_1287: 
        bReverseCarForward = false; 
        if (bAllowBackwardsDemolitions) { 
            bReverseCarForward = OldRBState.LinearVelocity Dot CarForward < 0; 
        } 
        if (PhysicsConfig.CarInteractionSettings.VictimHitAngleCheck.bEnabled) { 
            if (!IsCarHitAngleWithinForwardAngle(HitCar, Impact, PhysicsConfig.CarInteractionSettings.VictimHitAngleCheck.DemolishAngleYaw, PhysicsConfig.CarInteractionSettings.VictimHitAngleCheck.DemolishAnglePitch, bReverseCarForward)) { 
                Result = ECarImpactResult.CarImpactResult_FailNotWithinVictimHitLocationAngle; 
                return false; 
            } 
        } 
        if (PhysicsConfig.CarInteractionSettings.AttackerHitAngleCheck.bEnabled) { 
            if (!IsHitLocationWithinForwardAngle(Impact, PhysicsConfig.CarInteractionSettings.AttackerHitAngleCheck.DemolishAngleYaw, PhysicsConfig.CarInteractionSettings.AttackerHitAngleCheck.DemolishAnglePitch, bReverseCarForward)) { 
                Result = ECarImpactResult.CarImpactResult_FailNotWithinForwardHitAngle; 
                return false; 
            } 
        } 
        if (PhysicsConfig.CarInteractionSettings.VictimHitAngleCurveCheck.bEnabled) { 
            if (!IsCarHitAngleWithinForwardAngleCurve(HitCar, Impact, PhysicsConfig.CarInteractionSettings.VictimHitAngleCurveCheck.DemolishAngleCurveYaw, PhysicsConfig.CarInteractionSettings.VictimHitAngleCurveCheck.DemolishAngleCurvePitch, bReverseCarForward)) { 
                Result = ECarImpactResult.CarImpactResult_FailNotWithinVictimHitLocationAngle; 
                return false; 
            } 
        } 
        if (PhysicsConfig.CarInteractionSettings.COMAngleCheck.bEnabled) { 
            if (!IsCarWithinForwardEllipticalCone(HitCar, Impact, PhysicsConfig.CarInteractionSettings.COMAngleCheck.DemolishAngleYaw, PhysicsConfig.CarInteractionSettings.COMAngleCheck.DemolishAnglePitch, bReverseCarForward)) { 
                Result = ECarImpactResult.CarImpactResult_FailNotWithinForwardEllipticalCone; 
                return false; 
            } 
        } 
        if (PhysicsConfig.CarInteractionSettings.bCheckImpactNormal) { 
            if (!IsValidImpactNormalHit(Impact, PhysicsConfig.CarInteractionSettings.ImpactNormalDotProductDemo)) { 
                Result = ECarImpactResult.CarImpactResult_FailNotWithinImpactNormalAngle; 
                return false; 
            } 
        } 
    } 
    if (DemolishTarget == EDemolishTarget.DemolishTarget_OtherTeam) { 
        if (PlayerReplicationInfo.Team != None && PlayerReplicationInfo.Team == HitCar.PlayerReplicationInfo.Team) { 
            Result = ECarImpactResult.CarImpactResult_FailSameTeam; 
            return false; 
        } 
    } 
    Result = ECarImpactResult.CarImpactResult_Success; 
    return true; 
} 