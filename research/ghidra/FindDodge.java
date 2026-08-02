import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import java.util.*;

public class FindDodge extends GhidraScript {
    public void run() throws Exception {
        // UE3 keeps these in the FName table, not as defined string Data, so search raw bytes.
        String[] keys = {"DodgeTorque","DodgeImpulse","Dodge","DoubleJump","AirControl","Stall"};
        for (String k : keys) {
            Address a = null; int found = 0;
            println("=== " + k + " ===");
            while (found < 12) {
                a = find(a == null ? currentProgram.getMinAddress() : a.add(1), k.getBytes());
                if (a == null) break;
                found++;
                StringBuilder sb = new StringBuilder("  " + a);
                ReferenceIterator it = currentProgram.getReferenceManager().getReferencesTo(a);
                int nref = 0;
                while (it.hasNext() && nref < 4) {
                    Function f = getFunctionContaining(it.next().getFromAddress());
                    if (f != null) { sb.append("  <- ").append(f.getName())
                        .append("@").append(f.getEntryPoint()); nref++; }
                }
                println(sb.toString());
            }
            if (found == 0) println("  (none)");
        }
    }
}
